import os
from function import Function


class FunctionNode:
    """
    Data structure for storing components of functions
    """
    def __init__(self, base_fn: list=[], paren_children: list=[], bracket_children: list=[], brace_children: list=[], idx: int=None):
        self.base_fn = base_fn  # char list
        self.paren_children = paren_children
        self.bracket_children = bracket_children
        self.brace_children = brace_children
        # index in full tree's string
        self.idx = idx
    
    def get_base_fn(self):
        if type(self.base_fn) == list:
            self.base_fn = ''.join(self.base_fn).strip()
        return self.base_fn

    def get_fn_nochildren(self):
        bracket_substr = ','.join([str(children) for children in self.bracket_children])
        if len(bracket_substr) > 0:
            bracket_substr = f"[{bracket_substr}]"
        brace_substr = ','.join([str(children) for children in self.brace_children])
        if len(brace_substr) > 0:
            brace_substr = f"{{{brace_substr}}}"
        return f"{self.get_base_fn()}{brace_substr}{bracket_substr}"
    
    def __str__(self):
        paren_substr = ','.join([str(children) for children in self.paren_children])
        if len(paren_substr) > 0:
            paren_substr = f"({paren_substr})"
        return f"{self.get_fn_nochildren()}{paren_substr}"


PAREN_TYPES = {
    '(': ')',
    '{': '}',
    '[': ']',
}

def parse_to_function_tree(s):
    """
    get matching parens to parse function
    """
    if ',' not in s and '(' not in s and '[' not in s and '{' not in s:
        # base case: no inner functions
        return FunctionNode(full_func=s, fn=s)
    else:

        toret = {}
        pstack = [FunctionNode(base_fn=[], idx=0, paren_children=[], bracket_children=[], brace_children=[])]  # top of this is the current node that we're getting base name of
        depth = 0
        curr_context_paren_type = []  # which paren type are we currently in
        has_comma_in_curr_context = []
        for i, c in enumerate(s):
            if c in PAREN_TYPES:
                curr_context_paren_type.append(c)
                pstack.append(FunctionNode(base_fn=[], idx=i+1, paren_children=[], bracket_children=[], brace_children=[]))  #, context_paren_type=c))
                has_comma_in_curr_context.append(False)
            elif c in PAREN_TYPES.values() or c == ',':
                if len(pstack) == 0:
                    raise IndexError("No matching closing parens at: " + str(i))
                ret_node = pstack.pop()
                assert len(curr_context_paren_type) > 0
                curr_paren_type = curr_context_paren_type.pop()
                has_comma = has_comma_in_curr_context.pop()
                # Add `ret_node` to top node in stack (direct parent)
                if curr_paren_type == '(':
                    if len(pstack[-1].paren_children) > 0:
                        assert has_comma, "cannot nest functions yet!"
                    pstack[-1].paren_children.append(ret_node)
                elif curr_paren_type == '[':
                    pstack[-1].bracket_children.append(ret_node)
                elif curr_paren_type == '{':
                    pstack[-1].brace_children.append(ret_node)
                if c == ',':
                    has_comma_in_curr_context.append(True)
                    pstack.append(FunctionNode(base_fn=[], idx=i+1, paren_children=[], bracket_children=[], brace_children=[]))  #, context_paren_type=c))
                    curr_context_paren_type.append(curr_paren_type)
            else:
                pstack[-1].base_fn.append(c)

        assert len(pstack) == 1
        return pstack[0]


def make_sampler_from_function(fn_tree, sample_type="seq"):
    # parses `fn_string` into a sampling function that returns sets of words
    if sample_type == "seq":
        assert fn_tree.get_base_fn() in ["filter", "map"]
        # mapfilter
        return parse_mapfilter_fn(fn_tree)
    else:
        raise NotImplementedError


def convert_bool_to_set(fn_tree):
    # convert a bool function to a set function
    if len(fn_tree.paren_children) == 0:
        return fn_tree
    else:
        if fn_tree.get_base_fn() == "in":
            # get set that element has to be in
            fn_tree = fn_tree.paren_children[1]
        
        elif fn_tree.get_base_fn() == "or":
            fn_tree.base_fn = "union"
        
        elif fn_tree.get_base_fn() == "and":
            fn_tree.base_fn = "intersection"
        
        for p, paren_child in enumerate(fn_tree.paren_children):
            fn_tree.paren_children[p] = convert_bool_to_set(paren_child)
        
        return fn_tree


def get_parents_of_input_nodes(fn_tree, known_nodes={}):
    # iterate down to 2nd-to-last `generation` of tree
    # {2ndtolast node -> [idxs of its children that are input nodes]}
    parent_2_input_node_idxs_map = {}

    visited = []
    visited.append(fn_tree)
    # iterate through tree
    while len(visited) > 0:
        node = visited.pop()
        for p, paren_child in enumerate(node.paren_children):
            if len(paren_child.paren_children) == 0:
                # `node` a parent to an input node
                if node not in parent_2_input_node_idxs_map:
                    parent_2_input_node_idxs_map[node] = []
                parent_2_input_node_idxs_map[node].append(p)
            visited.append(paren_child)
    return parent_2_input_node_idxs_map


def get_input(fn_str):
    """
    Get input var to `fn_str`
    """
    # parse to innermost inputs
    if "(" not in fn_str:
        return {fn_str}
    else:
        inner_fns = []
        parens_stack = []
        for c, char in enumerate(fn_str):
            if char == '(':
                parens_stack.append(c)
            elif char == ')':
                start_idx = parens_stack.pop()
                if len(parens_stack) == 0:
                    inner_fns.append(fn_str[start_idx+1:c])
        # find matching closing
        # segment inner_fns
        all_inputs = set()
        for inner_fn in inner_fns:
            all_inputs = all_inputs.union(get_input(inner_fn))
        return all_inputs


def parse_mapfilter_fn(fn_tree):
    # condense compositions of map() & filter() into single mapfilter{<map_fn>, <filter_fn>}(0)
    all_map_fns = []
    all_filter_fns = []
    curr_node = fn_tree
    filter_input = None
    map_input = None
    while len(curr_node.paren_children) > 0:
        if curr_node.get_base_fn() not in ['filter', 'map']:
            import pdb; pdb.set_trace()
        if curr_node.get_base_fn() in ['filter']:
            all_filter_fns.append(curr_node.brace_children[0])
            filter_input = curr_node.brace_children[0].paren_children
        if curr_node.get_base_fn() in ['map']:
            all_map_fns.append(curr_node.brace_children[0])
            map_input = curr_node.brace_children[0].paren_children
        assert len(curr_node.paren_children) == 1
        curr_node = curr_node.paren_children[0]
    brace_children = [None, None]
    # compose all maps
    if len(all_map_fns) > 0:
        composed_maps = all_map_fns[0]
        # get parents of input nodes (to set them to `map_fn`)
        input_node_parents = get_parents_of_input_nodes(composed_maps)
        for i in range(1,len(all_map_fns)):
            for node in input_node_parents:
                for child_idx in input_node_parents[node]:
                    if str(node.paren_children[child_idx]) in get_input(str(all_map_fns[i])):
                        node.paren_children[child_idx] = all_map_fns[i]
            input_node_parents = get_parents_of_input_nodes(composed_maps)
        brace_children[0] = composed_maps
    # compose all filters (conjunction over all filters)
    if len(all_filter_fns) > 0:
        composed_filters = convert_bool_to_set(all_filter_fns[0])
        for i in range(1,len(all_filter_fns)):
            composed_filters = FunctionNode(base_fn='intersection', paren_children=[
                composed_filters, convert_bool_to_set(all_filter_fns[i]),
            ])
        brace_children[1] = composed_filters
    return FunctionNode(
        base_fn='mapfilter',
        paren_children=[curr_node],  # innermost child (input)
        brace_children=brace_children,
    )


def convert_relation_to_nl(relation: str=None, relation_tree=None, relation_fn=None):
    if relation_fn is None:
        if relation_tree is None:
            relation_tree = parse_to_function_tree(relation)
        relation_fn = Function.build(fn_tree=relation_tree, suppress_print=True, universe=None)
    return relation_fn.to_nl()


def test_function_parse():
    inputs = ["A[S](B,C)","A[S](B(D,E),C)","M{A[B](C(N))}","mapfilter{synonyms(X),has_pos[pos>eng](CONST(n))}","mapfilter{,has_pos[pos>eng](CONST(n))}"]
    for input in inputs:
        tree = parse_to_function_tree(input)
        print(str(tree))
        assert str(tree) == input

def test_mapfilter_flatten():
    inputs = [
        "map{A(0)}(0)", "filter{B(0)}(0)",
        "map{A(0)}(filter{B(0)}(0))", "filter{B(0)}(map{A(1)}(0))",
        "map{A(0)}(map{B(0)}(0))", "map{C(D(0))}(filter{B(0)}(map{A(1)}(0)))",
        "map{union(A(0),B(0))}(map{C(0)}(S))", "map{union(A(0),B(1))}(map{C(0)}(S))",
        "map{C(0)}(map{union(A(0),B(1))}(S))",
        "filter{A(0)}(filter{B(0)}(0))", "filter{C(0)}(map{B(0)}(filter{A(1)}(0)))",
        "filter{in[eng>eng](0, has_pos[eng>eng](CONST[n]))}(S)",
        "filter{or(in[eng>eng](0, has_pos[eng>eng](CONST[n])), in[eng>eng](0, has_pos[eng>eng](CONST[v])))}(filter{in[eng>eng](0, has_pos[eng>eng](CONST[a]))}(S))",
    ]
    for input in inputs:
        fn_tree = parse_to_function_tree(input)
        print(parse_mapfilter_fn(fn_tree))

def test_convert_to_nl():
    relations = [
        "wiki{mother(0)}", "wiki{father(mother(0))}",
        "wiki{or(father(1), mother(0))}", "wiki{and(father(1), mother(0))}",
        "wiki{or(father(mother(0)), mother(0))}", "wiki{and(father(mother(1)), mother(0))}", 
        "wiki{or(and(father(1), mother(0)), mother(father(1)))}", "wiki{and(or(father(mother(1)), mother(0)), father(mother(0)))}",
        "filter{in(0, has_pos[eng>eng](CONST{n}))}(S)", "map{synonyms[eng>eng](0)}(S)",
        "map{random[eng>eng]{0}(0)}(map{union(synonyms[eng>eng](0),entailments[eng>eng](1))}(S))",
        "filter{in(0, has_pos[eng>eng](CONST{n}))}(map{hypernyms[eng>eng](0)}(filter{in(0, has_sentiment[eng>eng](CONST{neutral}))}(S)))",
        # "filter{or(in[eng>eng](0, has_sentiment[eng>eng](CONST{positive})), in[eng>eng](0, has_pos[eng>eng](CONST{v})))}(filter{in[eng>eng](0, has_pos[eng>eng](CONST{a}))}(S))",
        "synonyms[eng>eng](0)", "hypernyms[eng>eng](synonyms[eng>eng](0))",
        "union(random[eng>eng](0), synonyms[eng>eng](0))",
        # "synonyms[eng>eng](has_pos[eng>eng](CONST{n}))",
        "in(0,has_pos[eng>eng](CONST{n}))",
        "in(0,has_sentiment[eng>eng](CONST{negative}))",
    ]
    for relation in relations:
        print(relation)
        nl_relation = convert_relation_to_nl(relation)
        print(nl_relation)

if __name__ == "__main__":
    # test_function_parse()
    # test_mapfilter_flatten()
    test_convert_to_nl()