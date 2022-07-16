"""
Tests whether code produces correct output for existing functions
"""

from scripts.make_data import make_word_function, make_seq_function, make_sampler_from_function,  recursive_convert_underscores
import argparse
import os
from glob import glob
from function import Function
from function.utils import parse_to_function_tree, recursive_apply_str_func
import json
from function.wikidata_fns import WikidataEntity
from tqdm import tqdm
import warnings
import random

# warnings.catch_warnings()

def recursive_lowercase(collection):
    # lowercase, recursively accessing collections
    return recursive_apply_str_func(collection, lambda x: x.lower())


def read_data_file(function_fn, is_seq=False, is_wiki=False):
    gt_output = open(os.path.join(function_fn, "all.jsonl")).readlines()
    gt_output_dict = {}
    for line in gt_output:
        line = json.loads(line)
        gt_keys = line.keys()
        del line["train_tgts"]
        if not is_seq and not is_wiki:
            line = {
                key: sorted(line[key]) if type(line[key]) == list else line[key]
                for key in sorted(gt_keys)
            }
        if "inner_fns" in line:
            for inner_fn in line["inner_fns"]:
                line["inner_fns"][inner_fn] = sorted(line["inner_fns"][inner_fn])
        gt_output_dict[json.dumps(line["inputs"])] = json.dumps(line)
        # gt_output_dict.add((line["inputs"], line["all_tgts"], line["inner_fns"], line["nl_R"], line["R"]))
    return gt_output_dict, gt_keys


def make_function(function_fn):
    function_nl = os.path.split(function_fn)[-1]
    print(function_nl)
    fn_tree = parse_to_function_tree(function_nl)
    # check
    assert str(fn_tree).replace(' ','') == function_nl.replace(' ',''), f"{str(fn_tree)} and {function_nl} differ!"
    function = Function.build(fn_tree=fn_tree, suppress_print=True)
    return fn_tree, function


def test_wordwise_wn(parent_dir):
    for function_fn in glob(f"{parent_dir}/*"):
        if "random" in function_fn: continue
        if "wiki" in function_fn: continue
        gt_output_dict, gt_entry_keys = read_data_file(function_fn)

        fn_tree, function = make_function(function_fn)
        exs = make_word_function(None, fn_tree, function)
        assert len(exs) == len(gt_output_dict)
        for ex in exs:
            if "train_tgts" in ex:
                del ex["train_tgts"]
            if "inner_fns" in ex and "inner_fns" not in gt_entry_keys:
                del ex["inner_fns"]
            else:
                ex["inner_fns"] = recursive_lowercase(ex["inner_fns"])
            ex["inputs"] = recursive_lowercase(ex["inputs"])
            ex["all_tgts"] = recursive_lowercase(ex["all_tgts"])
            ex = {
                key: sorted(ex[key]) if type(ex[key]) == list else ex[key]
                for key in sorted(ex.keys())
            }
            if "inner_fns" in ex:
                for inner_fn in ex["inner_fns"]:
                    ex["inner_fns"][inner_fn] = sorted(ex["inner_fns"][inner_fn])
            ex_str = json.dumps(ex)
            try:
                assert ex_str == gt_output_dict[json.dumps(ex["inputs"])]
            except:
                print(function_fn)
                print(gt_output_dict[json.dumps(ex["inputs"])])
                print(ex_str)
                breakpoint()


def test_seq_wn(parent_dir):
    for function_fn in glob(f"{parent_dir}/*"):
        if "random" in function_fn: continue
        if "wiki" in function_fn: continue
        gt_output_dict, _ = read_data_file(function_fn, is_seq=True)
        fn_tree, function = make_function(function_fn)

        for Xs in gt_output_dict:
            gt_output = gt_output_dict[Xs]
            gt_output = json.loads(gt_output)
            Xs = json.loads(Xs)
            fn_input = [{x} for x in Xs[0]]
            fn_input = recursive_apply_str_func(fn_input, lambda x: x.replace(" ", "_"))
            # first check function
            fn_seqs = function(fn_input)
            gt_Ys = [set(pertok_output) for pertok_output in gt_output['all_tgts']]
            Ys = recursive_convert_underscores(fn_seqs['out'])
            try:
                assert len(Ys) == len(gt_Ys)
                for tok_idx in range(len(Ys)):
                    assert Ys[tok_idx] == gt_Ys[tok_idx]
            except:
                print(function_fn)
                print(Ys)
                print(gt_Ys)
                print("bad index: " + tok_idx)
                breakpoint()

        # now check that the sampler works and is sensible
        sample_function = make_sampler_from_function(fn_tree, sample_type='seq')
        input_sampler = Function.build(fn_tree=sample_function, maxlen=8, suppress_print=True)
        for _ in range(5):
            Xs = input_sampler()
            assert len(Xs) == 8
            fn_seqs = function(Xs)
            try:
                if "filter" in function_fn:
                    assert len(fn_seqs['out']) == 4
                else:
                    assert len(fn_seqs['out']) == 8
                for tok_outs in fn_seqs['out']:
                    assert len(tok_outs) > 0
            except:
                print(str(fn_tree))
                print(Xs)
                print(tok_outs)


def test_wordwise_wiki(parent_dir):
    for function_fn in glob(f"{parent_dir}/*"):
        if "wiki" not in function_fn: continue
        gt_output_dict, _ = read_data_file(function_fn, is_wiki=True)
        fn_tree, function = make_function(function_fn)

        # do random sampling
        inputs_to_check = random.sample(gt_output_dict.keys(), 100)
        for Xs in tqdm(inputs_to_check):
            gt_output = gt_output_dict[Xs]
            gt_output = json.loads(gt_output)
            Xs = json.loads(Xs)
            # Xs = recursive_apply_str_func(Xs, lambda x: x.replace(" ", "_"))
            for x in Xs:
                x['Qid'] = x['Qid'].replace('q', 'Q')
            fn_input = [{(WikidataEntity.from_dict(Xs[0]),)}]
            # first check function
            fn_output = function(fn_input)
            try:
                if type(gt_output['all_tgts'][0]) == bool:
                    gt_Ys = set(gt_output['all_tgts'])
                    Ys = set(bool(int(item)) for item in fn_output['out'])
                else:
                    gt_Ys = recursive_lowercase(set(ent['Qid'] for ent in gt_output['all_tgts']))
                    Ys = recursive_lowercase(set(ent.to_dict()['Qid'] for ent in fn_output['out']))
            except:
                breakpoint()
            try:
                assert Ys == gt_Ys
            except:
                print(function_fn)
                print(Ys)
                print(gt_Ys)
                breakpoint()
    
        # now check sampler works and is sensible
        function.max_results = 6
        test_samples = function.get_samples()
        assert len(test_samples) > 0, f"Got {len(test_samples)} samples"
        test_samples = random.sample(test_samples, 6)
        for sample in test_samples:
            sample_output = function([{sample}])
            try:
                assert len(sample_output['out']) > 0
            except:
                breakpoint()


def test_seq_wiki(parent_dir):
    # TODO.....
    for function_fn in glob(f"{parent_dir}/*"):
        if "wiki" not in function_fn: continue
        gt_output_dict, _ = read_data_file(function_fn, is_seq=True, is_wiki=True)
        fn_tree, function = make_function(function_fn)

        # do random sampling
        inputs_to_check = random.sample(gt_output_dict.keys(), 100)
        for Xs in tqdm(inputs_to_check):
            gt_output = gt_output_dict[Xs]
            gt_output = json.loads(gt_output)
            Xs = json.loads(Xs)[0]
            fn_input = []
            for ent_id, ent_name in zip(Xs['Qid'], Xs['ent_name']):
                fn_input.append({(WikidataEntity.from_dict({'Qid': ent_id.replace("q", "Q"), 'ent_name': ent_name}),)})
                # x['Qid'] = recursive_lowercase(x['Qid'])
            # """
            # Xs = [list(xs)[0][0].to_dict() for xs in Xs]
            # Xs = {key: [xs[key] for xs in Xs] for key in Xs[0].keys()}
            # Ys = [[y.to_dict() for y in ys] for ys in Ys]
            # train_tgts = [random.choice(ys) for ys in Ys]
            # train_tgts = {key: [tgt[key] for tgt in train_tgts] for key in train_tgts[0].keys()}
            # """
            # fn_input = [{(WikidataEntity.from_dict(Xs[0]),)}]
            # first check function
            fn_output = function(fn_input)
            if type(gt_output['all_tgts'][0][0]) == bool:
                breakpoint()
            else:
                gt_Ys = recursive_lowercase(set(ent[0]['Qid'] for ent in gt_output['all_tgts']))
                Ys = recursive_lowercase(set(ent[0].to_dict()['Qid'] for ent in fn_output['out']))
            try:
                assert Ys == gt_Ys
            except:
                print(function_fn)
                print(Ys)
                print(gt_Ys)
                breakpoint()
    
        # now check sampler works and is sensible
        sample_function = make_sampler_from_function(fn_tree, sample_type='seq')
        input_sampler = Function.build(fn_tree=sample_function, maxlen=8, suppress_print=True)
        for _ in range(5):
            Xs = input_sampler()
            assert len(Xs) == 8
            fn_seqs = function(Xs)
            try:
                if "filter" in function_fn:
                    assert len(fn_seqs['out']) == 4
                else:
                    assert len(fn_seqs['out']) == 8
                for tok_outs in fn_seqs['out']:
                    assert len(tok_outs) > 0
            except:
                print(str(fn_tree))
                print(Xs)
                print(tok_outs)



def main(args):
    # test_wordwise_wn("TaskBenchData_orig/atomic")
    # for parent_dir in glob("TaskBenchData_orig/word_composite/*"):
    #     test_wordwise_wn(parent_dir)
    # for parent_dir in glob("TaskBenchData_orig/seq_composite/*"):
    #     test_seq_wn(parent_dir)
    test_wordwise_wiki("TaskBenchData_orig/atomic")
    # for parent_dir in glob("TaskBenchData_orig/word_composite/*"):
    #     test_wordwise_wiki(parent_dir)
    # for parent_dir in glob("TaskBenchData_orig/seq_composite/*"):
    #     test_seq_wiki(parent_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--")
    args = parser.parse_args()
    main(args)
