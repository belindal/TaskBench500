from abc import ABC, abstractmethod

FUNCTION_REGISTRY = {}

def get_func_class_by_name(func_name):
    return FUNCTION_REGISTRY[func_name.lower()]


class Function(ABC):

    def __init__(self, fn_tree, inner_fns=[], **kwargs):
        self.fn_tree = fn_tree
        self.inner_fns = inner_fns
    
    def __str__(self):
        return str(self.fn_tree)

    @classmethod
    def get_func_name(cls):
        return [cls.__name__.lower()]
    
    def compute_inner_fns(self, inputs):
        inner_fn_outputs = []
        for inner_fn in self.inner_fns:
            inner_fn_outputs.append(inner_fn(inputs)['out'])
        return inner_fn_outputs

    @classmethod
    def build(cls, fn_tree, suppress_print=False, **kwargs):
        if not suppress_print:
            print(str(fn_tree))
        function_name = fn_tree.get_base_fn()
        inner_fn_trees = fn_tree.paren_children
        # Recursively builds inner functions here (calls this function until no `(` left)
        inner_fns = [
            Function.build(fn_tree=inner_fn_tree, suppress_print=suppress_print, **kwargs)
            for inner_fn_tree in inner_fn_trees
        ]
        if 'inner_fns' in kwargs:
            del kwargs['inner_fns']
        if function_name == "is":
            assert len(fn_tree.brace_children) == 1
            attr_name = str(fn_tree.brace_children[0]).split('=')[0]
            function_class = FUNCTION_REGISTRY[attr_name]
        else:
            function_class = FUNCTION_REGISTRY[function_name]
        return function_class.build(fn_tree=fn_tree, inner_fns=inner_fns, suppress_print=suppress_print, **kwargs)

    def __init_subclass__(cls, **kwargs):
        """Register all children in registry"""
        super().__init_subclass__(**kwargs)
        func_names = cls.get_func_name()

        for func_name in func_names:
            assert (
                func_name not in FUNCTION_REGISTRY
            ), f"{func_name} task already registered!"
            FUNCTION_REGISTRY[func_name] = cls


ALL_VOCAB_FILES = {
    'eng': "resources/vocab_brown_5.tsv",
    'spa': "resources/vocab_cess_1.tsv",
}
def get_lang_vocab(lang):
    if lang is None:
        return None
    vocab = set()
    with open(ALL_VOCAB_FILES[lang]) as f:
        for line in f:
            vocab.add(line.strip())
    return vocab


class WordFunction(Function):
    """
    Functions on sets of words, including relations (word -> {words}) and predicates (word -> bool)
    """
    # cache of function -> {word -> {related words}}
    function_cache: dict = {}

    def __init__(self, fn_tree, inner_fns, **kwargs):
        super().__init__(fn_tree=fn_tree, inner_fns=inner_fns)
    
    def get_universe(self):
        if not hasattr(self, 'universe'):
            inner_fn_universe = None
            if self.fn_tree.get_base_fn() == "wiki":
                self.universe = self.domain()
            else:
                assert len(self.inner_fns) > 0, "Function should be overridden for input-returning function"
                for inner_fn in self.inner_fns:
                    if inner_fn_universe is None:
                        inner_fn_universe = inner_fn.get_universe()
                    else:
                        assert inner_fn_universe == inner_fn.get_universe()
                self.universe = inner_fn_universe
                if self.universe is None:
                    assert hasattr(self, 'lang')
                    self.universe = get_lang_vocab(self.lang)
        return self.universe

    def get_cached_words_results(self, inputs: list=None):
        assert len(inputs) == 1
        word_set = inputs[0]
        related_words = set()
        missing_word_set = set()
        for word in word_set:
            if WordFunction.function_cache.get(str(self)) is not None and word in WordFunction.function_cache.get(str(self)):
                related_words = related_words.union(WordFunction.function_cache[str(self)][word])
            else:
                missing_word_set.add(word)
        return related_words, missing_word_set

    def domain(self):
        if self.fn_tree.get_base_fn() == "wiki":
            # wiki
            domain = self.get_samples()
        else:
            # wordnet
            domain = set()
            for word in self.get_universe():
                all_related_words = self([{word}])
                if len(all_related_words['out']) > 0:
                    domain.add(word)
                    if WordFunction.function_cache.get(str(self)) is None: WordFunction.function_cache[str(self)] = {}
                    WordFunction.function_cache[str(self)][word] = all_related_words['out']
        return domain


from function.seq_ops import *
from function.set_ops import *
from function.logical_ops import *
from function.basic_atoms import *
from function.wordnet_atoms import *
from function.random_atoms import *
from function.wikidata_fns import *
from function.return_input import *