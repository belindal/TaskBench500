from function import Function, FUNCTION_REGISTRY, WordFunction


class LogicalAnd(WordFunction):
    """
    Maps bool-> bool
    """
    def __init__(self, fn_tree, inner_fns, **kwargs):
        super().__init__(fn_tree=fn_tree, inner_fns=inner_fns)
        assert len(self.inner_fns) <= 2

    @classmethod
    def get_func_name(cls):
        return ['land']

    def to_nl(self):
        inner_nls = [inner_fn.to_nl() for inner_fn in self.inner_fns]
        assert len(inner_nls) == 2
        return f"{inner_nls[0]} and {inner_nls[1]}"
    
    def __call__(self, inputs: list=None):
        inputs = self.compute_inner_fns(inputs)
        assert len(inputs) == 2
        return {'out': inputs[0] and inputs[1], 'inner': inputs}
    
    @classmethod
    def build(cls, fn_tree, inner_fns, **kwargs):
        return cls(fn_tree=fn_tree, inner_fns=inner_fns, **kwargs)


class LogicalOr(WordFunction):
    """
    Maps bool-> bool
    """
    def __init__(self, fn_tree, inner_fns, **kwargs):
        super().__init__(fn_tree=fn_tree, inner_fns=inner_fns)
        assert len(self.inner_fns) <= 2

    @classmethod
    def get_func_name(cls):
        return ['lor']

    def to_nl(self):
        inner_nls = [inner_fn.to_nl() for inner_fn in self.inner_fns]
        assert len(inner_nls) == 2
        return f"{inner_nls[0]} or {inner_nls[1]}"

    def __call__(self, inputs: list=None):
        inputs = self.compute_inner_fns(inputs)
        assert len(inputs) == 2
        return {'out': inputs[0] or inputs[1], 'inner': inputs}
    
    @classmethod
    def build(cls, fn_tree, inner_fns, **kwargs):
        return cls(fn_tree=fn_tree, inner_fns=inner_fns, **kwargs)


class LogicalNot(WordFunction):
    """
    Maps bool-> bool
    """
    def __init__(self, fn_tree, inner_fns, **kwargs):
        super().__init__(fn_tree=fn_tree, inner_fns=inner_fns)
        self.inner_fns = inner_fns

    @classmethod
    def get_func_name(cls):
        return ['lnot']

    def __call__(self, inputs: list=None):
        inputs = self.compute_inner_fns(inputs)
        assert len(inputs) == 1
        return {'out': not inputs[0], 'inner': inputs}
    
    @classmethod
    def build(cls, fn_tree, inner_fns, **kwargs):
        return cls(fn_tree=fn_tree, inner_fns=inner_fns, **kwargs)