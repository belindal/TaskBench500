# Print numbers in corresponding order as in input file
import json
from tqdm import tqdm
import os
from glob import glob
import numpy as np
import random
import argparse
import pandas as pd
import itertools
from run_models.utils import check_string_token_overlap


def get_metrics_from_jsonl(results_fn, possible_replacement_Ys: list=None, use_segmentation_char: bool=False):
    """
    Specify `possible_replacement_Ys` to subsitute Ys with random item from list.
    """
    accuracy = 0.0
    result_lines = []
    num_exs = 0
    with open(results_fn) as f:
        for line in f:
            num_exs += 1
            line = json.loads(line)
            if possible_replacement_Ys is not None:
                accuracy += random.choice(possible_replacement_Ys) in line.get('all_tgts', line['train_tgts'])
            else:
                if 'all_valid_gens' in line and type(line['all_valid_gens'][0]) == list:
                    if use_segmentation_char:
                        pred_tokens = [tok.strip() for tok in line['pred'].split('#')]
                        max_n_toks_correct = 0
                        for tidx in range(min(len(pred_tokens), len(line['all_valid_gens']))):
                            max_n_toks_correct += (pred_tokens[tidx] in line['all_valid_gens'][tidx])
                    else:
                        all_valid_gens_vocab = set().union(*line['all_valid_gens'])
                        line['all_valid_gen_vocab'] = set()
                        line['max_tok_len'] = 1
                        # add prefixes
                        for word in all_valid_gens_vocab:
                            if word.count(' ') > 1:
                                word_toks = word.split(' ')
                                prefixes = set([' '.join(word_toks[:w]) for w in range(2,len(word_toks))])
                                line['all_valid_gen_vocab'].union(prefixes)
                                line['max_tok_len'] = max(line['max_tok_len'], word.count(' ') + 1)
                            else:
                                line['all_valid_gen_vocab'].add(word)
                        max_n_toks_correct, _, _ = check_string_token_overlap(
                            line['pred'], line['all_valid_gens'],
                            all_possible_words=line['all_valid_gen_vocab'], max_tok_len=line['max_tok_len'],
                        )
                    
                    result_lines.append(line)
                    # accuracy += max_n_toks_correct == len(line['all_valid_gens'])
                    accuracy += float(max_n_toks_correct) / len(line['all_valid_gens'])
                else:
                    accuracy += line['pred'] in line.get('all_valid_gens', [line['gold']])
                    result_lines.append(line)
    if num_exs > 0:
        accuracy /= num_exs
    return accuracy, result_lines


def main(args):
    output_file = args.output_file
    data_dir = args.data_dir
    functions_file = args.functions_file
    use_segmentation_char = args.use_segmentation_char
    
    if not os.path.exists(output_file):
        with open(functions_file) as f:
            existing_function_results = pd.read_csv(f, sep='\t', header=0)
            for col in existing_function_results:
                if col != 'function':
                    existing_function_results[col] = np.nan
            functions = existing_function_results['function']
        train_types = existing_function_results.columns.tolist()
        train_types = [[tt+'_gen', tt+'_mem'] for tt in train_types if tt != 'function']
        train_types = [tt + [tt[0]+'_w_rel', tt[1]+'_w_rel'] if 'tune' in tt[0] else tt for tt in train_types]
        train_types = list(itertools.chain(*train_types))
    else:
        with open(output_file) as f:
            existing_function_results = pd.read_csv(f, sep='\t', header=0)
            functions = existing_function_results['function']
        train_types = existing_function_results.columns.tolist()
        train_types = [tt for tt in train_types if tt != 'function']

    get_splits = False
    get_accuracies = True
    wf = open(output_file, "w")
    if get_splits:
        for i, function in tqdm(enumerate(functions)):
            function = function.strip()
            dirname = data_dir.replace('{FXN}', function)
            if not os.path.exists(os.path.join(dirname, "all.jsonl")):
                wf.write(f"{function}\tInvalid\t{dirname}\n")
                wf.flush()
                continue
            if get_splits:
                split_amts = []
                for split in ['train', 'dev', 'test']:
                    split_amt = open(os.path.join(dirname, f"{split}.jsonl")).readlines()
                    split_amts.append(str(len(split_amt)))
                wf.write(f"{function}\t{'/'.join(split_amts)}\n")
                wf.flush()

    if get_accuracies:
        wf.write('function\t' + '\t'.join(train_types)+"\n")
        wf.flush()
        for i, function in tqdm(enumerate(functions)):
            function = function.strip()
            
            results = {}
            accuracies = {}
            function_fn_pattern = function.replace('[', '!@#$@').replace(']', '%&^%$')
            function_fn_pattern = function_fn_pattern.replace('!@#$@', '[[]').replace('%&^%$', '[]]')
            function_fn_pattern = data_dir.replace('{FXN}', function_fn_pattern)
            fns = {
                'rand_train_distr_gen': os.path.join(function_fn_pattern, 'dev.jsonl'),
                'likeliest_word_gen': os.path.join(function_fn_pattern, 'dev.jsonl'),
                'rand_train_distr_mem': os.path.join(function_fn_pattern, 'train.jsonl'),
                'likeliest_word_mem': os.path.join(function_fn_pattern, 'train.jsonl'),
                'Prompt-tune_full_gen': f"{function_fn_pattern.replace('data', 'checkpoints')}{'_segment' if use_segmentation_char else ''}/t5-base_lr1.0_seed[0-9]_prompt100.jsonl",
                'Finetune_full_gen': f"{function_fn_pattern.replace('data', 'checkpoints')}{'_segment' if use_segmentation_char else ''}/t5-base_lr0.001_seed[0-9].jsonl",
                'Prompt-tune_full_mem': f"{function_fn_pattern.replace('data', 'checkpoints')}{'_segment' if use_segmentation_char else ''}/t5-base_lr1.0_seed[0-9]_prompt100_mem.jsonl",
                'Finetune_full_mem': f"{function_fn_pattern.replace('data', 'checkpoints')}{'_segment' if use_segmentation_char else ''}/t5-base_lr0.001_seed[0-9]_mem.jsonl",
            }
            for ntrain in [10, 32, 100] + list(range(0, 1001, 250)):
                fns[f'Prompt-tune_{ntrain}_gen'] = f"{function_fn_pattern.replace('data', 'checkpoints')}{'_segment' if use_segmentation_char else ''}/t5-base_lr1.0_seed[0-9]_prompt100_fs{ntrain}.jsonl"
                fns[f'Finetune_{ntrain}_gen'] = f"{function_fn_pattern.replace('data', 'checkpoints')}{'_segment' if use_segmentation_char else ''}/t5-base_lr0.001_seed[0-9]_fs{ntrain}.jsonl"
                fns[f'Prompt-tune_{ntrain}_mem'] = f"{function_fn_pattern.replace('data', 'checkpoints')}{'_segment' if use_segmentation_char else ''}/t5-base_lr1.0_seed[0-9]_prompt100_fs{ntrain}_mem.jsonl"
                fns[f'Finetune_{ntrain}_mem'] = f"{function_fn_pattern.replace('data', 'checkpoints')}{'_segment' if use_segmentation_char else ''}/t5-base_lr0.001_seed[0-9]_fs{ntrain}_mem.jsonl"
            w_rel_fns = {}
            for fn in fns:
                if 'tune' in fn:
                    w_rel_fns[fn+"_w_rel"] = fns[fn].replace('t5-base_', 't5-base_w_rel_')
            fns = {**fns, **w_rel_fns}
            if not os.path.exists(os.path.join(data_dir.replace('{FXN}', function), 'all.jsonl')):
                import pdb; pdb.set_trace()
                wf.write(f"{function}\tInvalid\t{data_dir.replace('{FXN}', function)}\n")
                wf.flush()
                continue
            # compute word frequencies in train set for baselines
            if 'rand_train_distr_gen' in train_types or 'rand_train_distr_mem' in train_types or 'likeliest_word_gen' in train_types or 'likeliest_word_mem' in train_types:
                all_Ys = []
                Y2count = {}
                with open(os.path.join(data_dir.replace('{FXN}', function), 'train.jsonl')) as f:
                    for line in f:
                        try:
                            line = json.loads(line)
                        except:
                            import pdb; pdb.set_trace()
                        for train_tgt in line['train_tgts']:
                            all_Ys.extend(train_tgt)
                            # TODO if classify formulation...?
                            if type(train_tgt) == dict:
                                if train_tgt['ent_name'] not in Y2count:
                                    Y2count[train_tgt['ent_name']] = 0
                                Y2count[train_tgt['ent_name']] += 1
                            else:
                                if train_tgt not in Y2count:
                                    Y2count[train_tgt] = 0
                                Y2count[train_tgt] += 1
                if type(line['train_tgts'][0]) == dict:
                    all_Ys = sorted(all_Ys, key=lambda x: Y2count[x['ent_name']], reverse=True)
                else:
                    all_Ys = sorted(all_Ys, key=lambda x: Y2count[x], reverse=True)
                most_common_Y = all_Ys[0]
            for train_type in train_types:
                if existing_function_results is not None and train_type in existing_function_results.columns and pd.notnull(existing_function_results.iloc[i][train_type]):
                    continue
                results[train_type] = []
                accuracies[train_type] = []
                for results_fn in glob(fns[train_type]):
                    if '_seed' in results_fn:
                        seed = int(results_fn.split('_seed')[-1].split('_')[0].split('.')[0])
                    if train_type.startswith('rand_train_distr'): possible_replacement_Ys = all_Ys
                    elif train_type.startswith('likeliest_word'): possible_replacement_Ys = [most_common_Y]
                    else: possible_replacement_Ys = None
                    accuracy, result_lines = get_metrics_from_jsonl(
                        results_fn, possible_replacement_Ys=possible_replacement_Ys, use_segmentation_char=use_segmentation_char,
                    )
                    accuracies[train_type].append(accuracy)
                    results[train_type].append(result_lines)

            # print(f"TASK {task}")
            stdout = []
            for train_type in train_types:
                if existing_function_results is not None and train_type in existing_function_results.columns and pd.notnull(existing_function_results.iloc[i][train_type]):
                    stdout.append(existing_function_results.iloc[i][train_type])
                elif len(accuracies[train_type]) == 0:
                    stdout.append("")
                elif len(accuracies[train_type]) == 1:
                    stdout.append(f"{accuracies[train_type][0] * 100}%")
                else:
                    stdout.append(f"{np.mean(accuracies[train_type]) * 100}% +/- {np.std(accuracies[train_type]) * 100}%")
            wf.write(f"{function}\t"+'\t'.join(stdout)+"\n")
            wf.flush()

    wf.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="format string with {FXN} at location of function name, indicating location of data folder")
    parser.add_argument('--output_file', type=str, help="which file to write results to")
    parser.add_argument('--functions_file', type=str, default=None, help="file containing list of functions to output")
    parser.add_argument('--use_segmentation_char', action='store_true', help="for sequential functions, whether to score the version that explicitizes segmentation characters")
    args = parser.parse_args()

    main(args)
