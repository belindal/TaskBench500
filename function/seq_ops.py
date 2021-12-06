import functools
import random
from function import Function, FUNCTION_REGISTRY
from function.set_ops import SetComplement
from function.utils import FunctionNode
import json


class SeqSampleBuilder(Function):
    """
    Builds sequence samples that fit particular examples
    """
    def __init__(
        self, xlen: int, ylen: int, fn_tree: str,
        element_map_fn=None, element_map_fn_domain=None, element_filter_set=None, element_nofilter_set=None,
        **kwargs,
    ):
        super().__init__(fn_tree=fn_tree)
        self.element_map_fn = element_map_fn
        # returns set of elements in/out of the set
        self.element_filter_set = element_filter_set
        self.element_nofilter_set = element_nofilter_set
        self.map_domain = element_map_fn_domain
        self.xlen = xlen
        self.ylen = ylen
        # seq -> existing output
        self.function_cache = {}
    
    @classmethod
    def get_func_name(cls):
        return ['mapfilter']

    def __call__(self, *kwargs):
        """
        x: list of words
        """
        x = []
        # returns samples and their possible function outputs
        # linearize maps and filters
        while len(x) < self.ylen:
            nonfiltered_els = self.element_nofilter_set
            if nonfiltered_els is None:
                nonfiltered_els = self.map_domain
            else:
                nonfiltered_els = nonfiltered_els.intersection(self.map_domain)
            nonfiltered_el = random.choice(list(nonfiltered_els))
            x.append({nonfiltered_el})
        while len(x) < self.xlen:
            filtered_el = random.choice(list(self.element_filter_set))
            x.append({filtered_el})
        random.shuffle(x)
        return x
    
    @classmethod
    def build(cls, fn_tree, max_len=8, **kwargs):
        assert len(fn_tree.brace_children) == 2
        element_fn_tree = fn_tree.brace_children
        # either map or filter has an argument
        has_map = (element_fn_tree[0] is not None and len(element_fn_tree[0].get_base_fn()) > 0)
        has_filter = (element_fn_tree[1] is not None and len(element_fn_tree[1].get_base_fn()) > 0)
        assert has_map or has_filter

        xlen = max_len
        if has_map:
            element_map_fn = Function.build(fn_tree=element_fn_tree[0], **kwargs)
            element_map_fn_domain = element_map_fn.domain()
            universe = element_map_fn.get_universe()
        if has_filter:
            ylen = int(max_len/2)
            element_nofilter_fn = Function.build(fn_tree=element_fn_tree[1], **kwargs)
            universe = element_nofilter_fn.get_universe()
            element_nofilter_set = set()
            # make acceptable / inacceptable sets
            for word in universe:
                if list(element_nofilter_fn([{word}])['out'])[0]:
                    element_nofilter_set.add(word)
            element_filter_set = universe - element_nofilter_set
        if not has_map:
            element_map_fn = lambda x: {'out': x}
            element_map_fn_domain = universe
        if not has_filter:
            ylen = max_len
            element_nofilter_set = universe
            element_filter_set = set()
        return cls(
            xlen=xlen, ylen=ylen, fn_tree=fn_tree,
            element_map_fn=element_map_fn,
            element_map_fn_domain=element_map_fn_domain,
            element_filter_set=element_filter_set,
            element_nofilter_set=element_nofilter_set,
        )


class SeqFunction(Function):
    def __init__(self, fn_tree, element_fn, inner_fns, **kwargs):
        """
        element_fn: maps elements -> elements
        """
        super().__init__(fn_tree=fn_tree, inner_fns=inner_fns)
        self.element_fn = element_fn
        self.inner_fns = inner_fns
    
    @classmethod
    def build(cls, fn_tree, inner_fns, **kwargs):
        element_fn_tree = fn_tree.brace_children[0]
        element_map_fn = Function.build(fn_tree=element_fn_tree, **kwargs)
        return cls(
            fn_tree=fn_tree,
            element_fn=element_map_fn,
            inner_fns=inner_fns,
        )


class Map(SeqFunction):
    @classmethod
    def get_func_name(cls):
        return ['map']
    
    def to_nl(self):
        assert len(self.inner_fns) == 1
        inner_nl = self.inner_fns[0].to_nl()
        if inner_nl == "%S%":
            inner_nl = "word in %S%"
        element_nl = self.element_fn.to_nl().split('%')[0].strip()
        if "of each" in inner_nl:
            return f"{element_nl} {inner_nl}"
        return f"{element_nl} each {inner_nl}"

    def __call__(self, seq):
        seq = self.compute_inner_fns(seq)
        assert len(seq) == 1
        return {'out': [self.element_fn([el])['out'] for el in seq[0]], 'inner': seq}


class Filter(SeqFunction):
    @classmethod
    def get_func_name(cls):
        return ['filter']
    
    def to_nl(self):
        assert len(self.inner_fns) == 1
        inner_nl = self.inner_fns[0].to_nl()
        if inner_nl == "%S%":
            inner_nl = "word in %S%"
        element_nl = self.element_fn.to_nl().split('include')[0].strip()
        if "for which" in inner_nl:
            return f"{inner_nl} and {element_nl}"
        return f"{inner_nl} for which {element_nl}"

    def __call__(self, seq):
        seq = self.compute_inner_fns(seq)
        assert len(seq) == 1
        return {'out': [el for el in seq[0] if True in self.element_fn([el])['out']], 'inner': seq}
