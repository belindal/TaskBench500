from function import Function, FUNCTION_REGISTRY

class ReturnInput(Function):
    """
    Returns subset of input as specified by `X`, `Y`, or index.
    If input has quotes, returns a constant.
    """
    def __init__(self, fn_tree, index: int = None, constant: str = None):
        super().__init__(fn_tree=fn_tree, inner_fns=[])
        self.index = index
        self.constant = constant
    
    @classmethod
    def get_func_name(cls):
        # take up to 10 inputs
        return ['CONST', 'X', 'Y', 'S'] + [str(i) for i in range(100)]
    
    def get_universe(self):
        return None

    def to_nl(self):
        if self.constant is not None:
            return self.constant
        else:
            return f"%{self.index}%"
    
    def __call__(self, inputs):
        if self.constant is not None:
            return {'out': {self.constant}}
        elif self.index == 'S':
            return {'out': inputs}
        else:
            return {'out': inputs[self.index]}

    @classmethod
    def build(cls, fn_tree, **kwargs):
        index = None
        constant = None
        if fn_tree.get_base_fn() == 'CONST':
            constant = fn_tree.brace_children[0].get_base_fn()
        else:
            if str(fn_tree.get_base_fn()) == 'X':
                index = 0
            elif str(fn_tree.get_base_fn()) == 'Y':
                index = 1
            elif str(fn_tree.get_base_fn()) == 'S':
                # return entire sequence
                index = 'S'
            else:
                index = int(str(fn_tree.get_base_fn()))
        return cls(fn_tree=fn_tree, index=index, constant=constant)