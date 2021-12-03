import json
import os
from function import Function, FUNCTION_REGISTRY, WordFunction
from function import get_lang_vocab


class AllWordPredicate(WordFunction):
    """
    Maps word -> true
    X -> true
    """
    def __init__(self, fn_tree, function, inner_fns, lang, **kwargs):
        super().__init__(fn_tree=fn_tree, inner_fns=inner_fns)

        # Getting pre-computed relations
        self.function = function
        self.lang = lang

    @classmethod
    def get_func_name(cls):
        return ['acceptall']

    def get_universe(self):
        # TODO
        if not hasattr(self, 'universe'):
            self.universe = super().get_universe()
            if self.universe is None:
                self.universe = get_lang_vocab(self.input_lang)
        return self.universe

    def to_nl(self):
        return str(True)

    def __call__(self, inputs: list=None):
        """
        get all words lexically related to any words in word_set
        """
        assert len(inputs) == 1
        return {'out': True, 'inner': self.compute_inner_fns(inputs)}

    @classmethod
    def build(cls, fn_tree, inner_fns, **kwargs):
        curr_function = fn_tree.get_base_fn()
        function = str(fn_tree)
        lang = fn_tree.bracket_children[0]

        return cls(
            fn_tree=fn_tree,
            function=curr_function,
            inner_fns=inner_fns,
            lang=lang,
            **kwargs,
        )

class NoWordPredicate(WordFunction):
    """
    Maps word -> false
    X -> false
    """
    def __init__(self, fn_tree, function, inner_fns, lang, **kwargs):
        super().__init__(fn_tree=fn_tree, inner_fns=inner_fns)

        # Getting pre-computed relations
        self.function = function
        self.lang = lang

    @classmethod
    def get_func_name(cls):
        return ['rejectall']

    def get_universe(self):
        # TODO
        if not hasattr(self, 'universe'):
            self.universe = super().get_universe()
            if self.universe is None:
                self.universe = get_lang_vocab(self.input_lang)
        return self.universe

    def to_nl(self):
        return str(False)

    def __call__(self, inputs: list=None):
        """
        get all words lexically related to any words in word_set
        """
        assert len(inputs) == 1
        return {'out': False, 'inner': self.compute_inner_fns(inputs)}

    @classmethod
    def build(cls, fn_tree, inner_fns, **kwargs):
        curr_function = fn_tree.get_base_fn()
        function = str(fn_tree)
        lang = fn_tree.bracket_children[0]

        return cls(
            fn_tree=fn_tree,
            function=curr_function,
            inner_fns=inner_fns,
            lang=lang,
            **kwargs,
        )

class WordIdentityMapping(WordFunction):
    """
    Maps word -> singleton set of a same word
    X -> {X}
    """
    def __init__(self, fn_tree, function, inner_fns, lang, **kwargs):
        super().__init__(fn_tree=fn_tree, inner_fns=inner_fns)

        # Getting pre-computed relations
        self.function = function
        self.lang = lang

    @classmethod
    def get_func_name(cls):
        return ['identity']

    def get_universe(self):
        # TODO
        if not hasattr(self, 'universe'):
            self.universe = super().get_universe()
            if self.universe is None:
                self.universe = get_lang_vocab(self.input_lang)
        return self.universe

    def to_nl(self):
        inner_nls = [inner_fn.to_nl() for inner_fn in self.inner_fns]
        assert len(inner_nls) == 1
        inner_nl = inner_nls[0]
        return f"same word as {inner_nl}".strip()

    def __call__(self, inputs: list=None):
        """
        get all words lexically related to any words in word_set
        """
        assert len(inputs) == 1
        return {'out': inputs[0], 'inner': self.compute_inner_fns(inputs)}

    @classmethod
    def build(cls, fn_tree, inner_fns, **kwargs):
        curr_function = fn_tree.get_base_fn()
        function = str(fn_tree)
        lang = fn_tree.bracket_children[0]

        return cls(
            fn_tree=fn_tree,
            function=curr_function,
            inner_fns=inner_fns,
            lang=lang,
            **kwargs,
        )