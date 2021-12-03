from tqdm import tqdm
import nltk
import functools
import argparse
import json
from glob import glob
import os
import time
from function import Function
import random
from function.utils import parse_to_function_tree, make_sampler_from_function, convert_relation_to_nl


def load_functions(function_file, functions):
    if function_file:
        with open(function_file) as f:
            functions = []
            for ln, line in enumerate(f):
                if line.split('\t')[0] == "function": continue
                functions.append(line.split('\t')[0].strip())
    print("Loaded functions")
    return functions


def make_word_function(args, fn_tree, function, wf):
    """
    get words
    """
    all_samples = function.domain()
    for w, word in tqdm(enumerate(all_samples)):
        fn_words = function([{word}])
        related_words = fn_words['out']
        inner_fn_words = fn_words['inner']
        if type(related_words) == bool:
            related_words = [related_words]
        if len(related_words) == 0:  # skip trivial cases
            continue
        ex = {}
        if fn_tree.get_base_fn() == 'wiki':
            for inp_num in range(len(word)):
                ex['inputs'].append([word[inp_num].to_dict()])
            if function.is_predicate:
                related_words = [int(related_word) == 1 for related_word in related_words]
            else:
                related_words = [w.to_dict() for w in related_words]
            ex['all_tgts'] = related_words
            ex['inner_fns'] = {fn: [int(inner_word) == 1 if inner_word in ['0','1'] else inner_word.to_dict() for inner_word in inner_fn_words[fn]] for fn in inner_fn_words}
        else:
            ex['inputs'] = [word]
            ex['all_tgts'] = list(related_words)
            ex['inner_fns'] = {str(fn): [inner_fn_words[f]] if type(inner_fn_words[f]) == bool else list(inner_fn_words[f]) for f, fn in enumerate(function.inner_fns)}
        ex["nl_R"] = convert_relation_to_nl(relation_fn=function)
        ex["R"] = str(function)
        wf.write(f"{json.dumps(ex)}\n")
        wf.flush()


def make_seq_function(args, fn_tree, function, wf):
    """
    get sequences
    """
    sample_function = make_sampler_from_function(fn_tree, sample_type='seq')
    input_sampler = Function.build(fn_tree=sample_function, maxlen=8, suppress_print=True)
    for s in tqdm(range(args.num_samples)):
        Xs = input_sampler()
        fn_seqs = function(Xs)
        Ys = fn_seqs['out']
        inner_fn_outs = fn_seqs['inner']
        Xs = [list(xs)[0] for xs in Xs]
        Ys = [list(ys) for ys in Ys]
        wf.write(json.dumps({
            "inputs": [Xs], "train_tgts": [[random.choice(ys) for ys in Ys]], "all_tgts": Ys,
            "inner_fns": {str(fn): [list(inner_fn_out) for inner_fn_out in inner_fn_outs[f]] for f, fn in enumerate(function.inner_fns)},
            "R": str(function), "nl_R": convert_relation_to_nl(relation_fn=function),
        })+"\n")
        wf.flush()


def make_functions(args, functions):
    skipped_fns = []
    for f, function_nl in enumerate(functions):
        function_savefile = f"{args.save_dir}/{function_nl}.jsonl"
        if os.path.exists(function_savefile):
            continue

        print(f"Generating data for function {f}/{len(functions)}: {function_nl}")
        print(f"Writing to {function_savefile}")
        wf = open(function_savefile, "w")
        fn_tree = parse_to_function_tree(function_nl)
        # check
        assert str(fn_tree).replace(' ','') == function_nl.replace(' ',''), f"{str(fn_tree)} and {function_nl} differ!"
        function = Function.build(fn_tree=fn_tree, suppress_print=True)

        if args.sample_type == 'seq':
            make_seq_function(args, fn_tree, function, wf)
    
        elif args.sample_type == 'word':
            make_word_function(args, fn_tree, function, wf)

        else:
            raise NotImplementedError
        wf.close()


def main(args):
    functions = load_functions(args.function_file, args.function)

    skipped_fns = make_functions(args, functions, make_splits=True)

    print(f"Skipped functions: {skipped_fns}")
    for fn in skipped_fns:
        os.remove(fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', default=None, help="stringified function")
    parser.add_argument('--function_file', default=None, help="file to list of stringified functions")
    parser.add_argument('--save_dir', default=None, help="directory to save functions in")
    parser.add_argument('--sample_type', choices=['seq', 'word'], help="type of inputs to sample")
    parser.add_argument('--num_samples', type=int, default=2000, help="number of sequences to sample")
    args = parser.parse_args()
    main(args)