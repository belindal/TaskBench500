import json
import os
from function import Function, FUNCTION_REGISTRY, WordFunction
from random import Random
from function import get_lang_vocab


class GetRandomRelatedWords(WordFunction):
    """
    Maps word -> singleton set of a random word
    X -> {Y}
    """
    def __init__(self, fn_tree, function, inner_fns, lang, randseed=0, **kwargs):
        super().__init__(fn_tree=fn_tree, inner_fns=inner_fns)

        # Getting pre-computed relations
        self.function = function
        self.lang = lang
        self.rand = Random(randseed)
        # cache, for this relation only (excluding inner functions)
        self.curr_relation_cache = {}

    @classmethod
    def get_func_name(cls):
        return ['random']

    def get_universe(self):
        if not hasattr(self, 'universe'):
            self.universe = get_lang_vocab(self.lang)
        return self.universe

    def to_nl(self):
        inner_nls = [inner_fn.to_nl() for inner_fn in self.inner_fns]
        assert len(inner_nls) == 1
        inner_nl = inner_nls[0]
        return f"random word to {inner_nl}".strip()

    def __call__(self, inputs: list=None):
        """
        get all words lexically related to any words in word_set
        """
        related_words, missing_word_set = super().get_cached_words_results(inputs)
        missing_word_set = self.compute_inner_fns([missing_word_set])
        if len(missing_word_set) == 0:
            return {'out': related_words, 'inner': self.compute_inner_fns(inputs)}
        assert len(missing_word_set) == 1, f"Random function can only have 1 inner functions!: Currently {str(self)} has {len(missing_word_set)}"
        missing_word_set = missing_word_set[0]
        for word in missing_word_set:
            # iterate through words, computing appropriate relation
            if word in self.curr_relation_cache:
                related_words = related_words.union(self.curr_relation_cache[word])
                continue
            
            # NOTE only gets here and generates new random if doesn't already have an associated word
            related_word = self.rand.choice(list(self.get_universe()))
            curr_word_related_set = {related_word}
            self.curr_relation_cache[word] = curr_word_related_set
            related_words = related_words.union(curr_word_related_set)

        return {'out': related_words, 'inner': self.compute_inner_fns(inputs)}

    @classmethod
    def build(cls, fn_tree, inner_fns, **kwargs):
        curr_function = fn_tree.get_base_fn()
        function = str(fn_tree)
        lang = str(fn_tree.bracket_children[0])
        randseed = int(fn_tree.brace_children[0].get_base_fn())

        return cls(
            fn_tree=fn_tree,
            function=curr_function,
            inner_fns=inner_fns,
            lang=lang,
            randseed=randseed,
            **kwargs,
        )