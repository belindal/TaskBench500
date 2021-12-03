from function import Function, FUNCTION_REGISTRY, WordFunction


class SetIntersect(WordFunction):
    """
    Maps set -> set
    """
    def __init__(self, fn_tree, inner_fns, **kwargs):
        super().__init__(fn_tree=fn_tree, inner_fns=inner_fns)
        assert len(self.inner_fns) <= 2

    @classmethod
    def get_func_name(cls):
        return ['intersection']

    def to_nl(self):
        inner_nls = [inner_fn.to_nl() for inner_fn in self.inner_fns]
        assert len(inner_nls) == 2
        return f"{inner_nls[0]} and {inner_nls[1]}"
    
    def __call__(self, inputs: list=None):
        inputs = self.compute_inner_fns(inputs)
        assert len(inputs) == 2
        return {'out': set(inputs[0]).intersection(set(inputs[1])), 'inner': inputs}
    
    @classmethod
    def build(cls, fn_tree, inner_fns, **kwargs):
        return cls(fn_tree=fn_tree, inner_fns=inner_fns, **kwargs)


class SetUnion(WordFunction):
    """
    Maps set -> set
    """
    def __init__(self, fn_tree, inner_fns, **kwargs):
        super().__init__(fn_tree=fn_tree, inner_fns=inner_fns)
        assert len(self.inner_fns) <= 2

    @classmethod
    def get_func_name(cls):
        return ['union']

    def to_nl(self):
        inner_nls = [inner_fn.to_nl() for inner_fn in self.inner_fns]
        assert len(inner_nls) == 2
        return f"{inner_nls[0]} or {inner_nls[1]}"

    def __call__(self, inputs: list=None):
        inputs = self.compute_inner_fns(inputs)
        assert len(inputs) == 2
        return {'out': set(inputs[0]).union(set(inputs[1])), 'inner': inputs}
    
    @classmethod
    def build(cls, fn_tree, inner_fns, **kwargs):
        return cls(fn_tree=fn_tree, inner_fns=inner_fns, **kwargs)


class SetComplement(WordFunction):
    """
    Maps set -> set
    """
    def __init__(self, fn_tree, inner_fns, **kwargs):
        super().__init__(fn_tree=fn_tree, inner_fns=inner_fns)
        self.inner_fns = inner_fns

    @classmethod
    def get_func_name(cls):
        return ['complement']

    def __call__(self, inputs: list=None):
        breakpoint()
        inputs = self.compute_inner_fns(inputs)
        assert len(inputs) == 1
        return {'out': (self.get_universe() - set(inputs[0])), 'inner': inputs}
    
    @classmethod
    def build(cls, fn_tree, inner_fns, **kwargs):
        return cls(fn_tree=fn_tree, inner_fns=inner_fns, **kwargs)


class Subset(Function):
    """
    Maps [set, set] -> bool
    """
    def __init__(self, inner_fns, **kwargs):
        self.inner_fns = inner_fns

    @classmethod
    def get_func_name(cls):
        return ['subset']

    def __call__(self, inputs: list = None):
        inputs = self.compute_inner_fns(inputs)
        assert len(inputs) == 2
        return {'out': {set(inputs[0]).issubset(set(inputs[1]))}, 'inner': inputs}

    @classmethod
    def build(cls, fn_tree, inner_fns, **kwargs):
        return cls(inner_fns)


class SetEquals(Function):
    """
    Maps [set, set] -> bool
    """
    def __init__(self, inner_fns, **kwargs):
        self.inner_fns = inner_fns

    @classmethod
    def get_func_name(cls):
        return ['eq']

    def __call__(self, inputs: list = None):
        inputs = self.compute_inner_fns(inputs)
        assert len(inputs) == 2
        return {'out': {set(inputs[0]) == set(inputs[1])}, 'inner': inputs}

    @classmethod
    def build(cls, fn_tree, inner_fns, **kwargs):
        return cls(inner_fns)


class IsIn(Function):
    """
    Maps [element, set] -> bool
    """
    def __init__(self, fn_tree, inner_fns, **kwargs):
        super().__init__(fn_tree=fn_tree, inner_fns=inner_fns)

    @classmethod
    def get_func_name(cls):
        return ['in']

    def to_nl(self):
        inner_nls = [inner_fn.to_nl() for inner_fn in self.inner_fns]
        assert len(inner_nls) == 2
        return f"{inner_nls[1]} include {inner_nls[0]}"

    def __call__(self, inputs: list = None):
        inputs = self.compute_inner_fns(inputs)
        assert len(inputs) == 2
        assert len(inputs[0]) == 1
        for el in inputs[0]: break
        return {'out': el in inputs[1], 'inner': inputs}

    @classmethod
    def build(cls, fn_tree, inner_fns, **kwargs):
        return cls(fn_tree=fn_tree, inner_fns=inner_fns)


class IsEmpty(Function):
    """
    Maps set -> bool
    """
    def __init__(self, inner_fns, **kwargs):
        self.inner_fns = inner_fns

    @classmethod
    def get_func_name(cls):
        return ['empty']

    def __call__(self, inputs: list=None):
        inputs = self.compute_inner_fns(inputs)
        assert len(inputs) == 1
        return {'out': len(inputs[0]) == 0, 'inner': inputs}

    @classmethod
    def build(cls, fn_tree, inner_fns, **kwargs):
        return cls(inner_fns)
