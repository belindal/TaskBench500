import os
import pandas as pd
import random
import csv
import argparse
from scripts.print_numbers_corresponding_order import get_metrics_from_jsonl
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='t5-base', help="Model architecture")
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--data_dir', type=str, default='data', help="Data directory")
parser.add_argument('--data_formulation', type=str, default='generate', choices=['generate', 'generate_w_rel'])
parser.add_argument('--do_prompt_tune', action='store_true', default=False, help="True to do prompt-tuning, else do full finetuning")
parser.add_argument('--num_fewshot', type=int, default=-1, help="Number of fewshot examples to use, set to -1 for full training data")
parser.add_argument('--n_prefix', type=int, default=100, help="Number of prefix tuning tokens")
parser.add_argument('--task_results_file', type=str, help="File containing all tasks and their state of completion")
parser.add_argument('--metric', type=str, default='infer', choices=['accuracy', 'token_accuracy', 'infer'], help="If `infer`, infers from whether `seq` appears in task name (`data_dir`)")
parser.add_argument('--chunk_size', type=int, default=5, help="# of tasks to complete at a time")
parser.add_argument('--no_pretrain', action='store_true', default=False, help="Use non-pretrained models")
parser.add_argument('--test_memorization_only', action='store_true', default=False, help="True to do only test memorization")
parser.add_argument('--continue_train', action='store_true', default=False, help="True to continue training")
parser.add_argument('--eval_only', action='store_true', default=False, help="True to only evaluate")
parser.add_argument('--use_segmentation_char', action='store_true', default=False, help="for sequences, whether to have a segmentation character in outputs")
parser.add_argument('--permute_labels', action='store_true', default=False, help="Train/evaluate on permuted labels")
args = parser.parse_args()

arch = args.arch
batchsize = args.batchsize
data_formulation = args.data_formulation
eval_batchsize = batchsize * 2
metric = args.metric
if metric == 'infer':
    if 'seq' in args.data_dir:
        metric = 'token_accuracy'
    else:
        metric = 'accuracy'
do_finetune = not args.do_prompt_tune
n_fs_examples = args.num_fewshot  # set to -1 if no few-shot

results_key = f'{"Finetune" if do_finetune else "Prompt-tune"}_{"full" if n_fs_examples == -1 else n_fs_examples}_{"mem" if args.test_memorization_only else "gen"}'


with open(args.task_results_file) as f:
    existing_results = pd.read_csv(f, sep='\t', header=0)
# choose a missing file
missing_rels = existing_results[results_key][existing_results[results_key].isnull()].index
if args.chunk_size < len(missing_rels):
    chosen_rel_idxs = random.sample(missing_rels.tolist(), k=args.chunk_size)
else:
    chosen_rel_idxs = missing_rels.tolist()
existing_results.loc[chosen_rel_idxs,results_key] = 'R'
with open(args.task_results_file, "w") as wf:
    existing_results.to_csv(wf, quoting=csv.QUOTE_NONE, sep='\t', header=True, index=False)
chosen_rels = existing_results.loc[chosen_rel_idxs,'function'].tolist()

save_dir = args.data_dir.replace('data', 'checkpoints')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

idx = 0
while len(missing_rels) > 0:
    rel2results = {}
    for rel in chosen_rels:
        rel2results[rel] = {'mem': [], 'gen': []}
        file = os.path.join(args.data_dir, rel)
        print(file)
        if not os.path.exists(os.path.join(file, "all.jsonl")):
            print("INVALID FILE")
            idx += 1
            continue
        additional_args = []
        if "map" in rel or "filter" in rel:
            # dealing with sequences...
            batchsize = 32
        else:
            batchsize = 64
        eval_batchsize = batchsize * 2
        if n_fs_examples > -1 and n_fs_examples <= 100:
            epochs = 100
            patience = 100
            eval_interval = 1
            NUM_TRIALS = 3
        else:
            epochs = 100
            patience = 50
            eval_interval = 1
            NUM_TRIALS = 1
        if do_finetune:
            # if n_fs_examples > -1 and n_fs_examples <= 100:
            #     epochs = 100
            #     patience = 100
            #     eval_interval = 1
            #     NUM_TRIALS = 4
            # else:
            #     epochs = 100
            #     patience = 50
            #     eval_interval = 1
            #     NUM_TRIALS = 1
            lr = 0.001
        else:
            # epochs = max(min(10000, int(5000000/n_fs_examples)), 150)
            # patience = max(min(500, int(500000/n_fs_examples)), 100)
            # eval_interval = max(min(50, int(5000/n_fs_examples)), 5)
            # NUM_TRIALS = min(max(1,int(4/np.log10(n_fs_examples))), 5)
            lr = 1.0
            num_prefixes = args.n_prefix
            additional_args.extend(["--do_prompt_tune", "--n_prefix", str(args.n_prefix)])
        if n_fs_examples > -1:
            additional_args.extend(["--train_size", n_fs_examples])
        if args.test_memorization_only or rel.startswith('random'):
            additional_args.extend(["--test_memorization_only"])
        if args.no_pretrain:
            additional_args.extend(["--no_pretrain"])
        if args.use_segmentation_char:
            additional_args.extend(["--use_segmentation_char"])
        if args.permute_labels:
            additional_args.extend(["--permute_labels"])
        
        for seed in range(NUM_TRIALS):
            print("\n=====")
            print(chosen_rels)
            print(f"{idx}/{len(missing_rels)}: {rel} - seed {seed}")
            print(file)
            print("=====\n")

            fp_end = file.split('/')[1:]
            fp_end[-1] += ("_segment" if args.use_segmentation_char else "")
            rel_dir = os.path.join(save_dir, *fp_end)
            os.makedirs(rel_dir, exist_ok=True)
            if do_finetune:
                mem_save_file = f"{rel_dir}/{arch}{data_formulation.replace('generate', '')}_lr{lr}_seed{seed}{f'_fs{n_fs_examples}' if n_fs_examples > -1 else ''}_mem.jsonl"
            else:
                mem_save_file = f"{rel_dir}/{arch}{data_formulation.replace('generate', '')}_lr{lr}_seed{seed}_prompt{num_prefixes}{f'_fs{n_fs_examples}' if n_fs_examples > -1 else ''}_mem.jsonl"
            save_file = mem_save_file.replace("_mem", "")
            if os.path.exists(mem_save_file) and not args.continue_train:
                rel2results[rel]['mem'].append(get_metrics_from_jsonl(mem_save_file)[0])
                if "--test_memorization_only" in additional_args: continue
            if os.path.exists(save_file) and not args.continue_train:
                rel2results[rel]['gen'].append(get_metrics_from_jsonl(save_file)[0])
                if os.path.exists(mem_save_file): continue
            if not args.eval_only:
                srun python run_models/ft_model.py \
                    --data @(file) --data_formulation @(data_formulation) \
                    --batchsize @(batchsize) --eval_batchsize @(eval_batchsize) \
                    --arch @(arch) --lr @(lr) \
                    --epochs @(epochs) --patience @(patience) \
                    --eval_interval @(eval_interval) \
                    --valid_metric @(metric) \
                    --seed @(seed) @(additional_args)
            srun python run_models/ft_model.py \
                --data @(file) --data_formulation @(data_formulation) \
                --batchsize @(batchsize) --eval_batchsize @(eval_batchsize) \
                --arch @(arch) --lr @(lr) \
                --valid_metric @(metric) \
                --seed @(seed) @(additional_args) \
                --eval_only
            if "--test_memorization_only" not in additional_args:
                srun python run_models/ft_model.py \
                    --data @(file) --data_formulation @(data_formulation) \
                    --arch @(arch) --lr @(lr) \
                    --valid_metric @(metric) \
                    --seed @(seed) @(additional_args) \
                    --test_memorization_only --eval_only
                if os.path.exists(save_file):
                    rel2results[rel]['gen'].append(get_metrics_from_jsonl(save_file)[0])
            if os.path.exists(mem_save_file):
                rel2results[rel]['mem'].append(get_metrics_from_jsonl(mem_save_file)[0])
        idx += 1

    with open(args.task_results_file) as f:
        existing_results = pd.read_csv(f, sep='\t', header=0)
    for rel in rel2results:
        for memgen in rel2results[rel]:
            memgen_results_key = f"{results_key.split('_')[0]}_{memgen}"
            # mark old results
            if len(rel2results[rel][memgen]) == 1:
                existing_results.loc[existing_results['function']==rel,memgen_results_key] = f"{rel2results[rel][memgen][0] * 100}%"
            elif len(rel2results[rel][memgen]) > 1:
                existing_results.loc[existing_results['function']==rel,memgen_results_key] = f"{np.mean(rel2results[rel][memgen]) * 100}% +/- {np.std(rel2results[rel][memgen]) * 100}%"
    # choose a missing file
    missing_rels = existing_results[results_key][existing_results[results_key].isnull()].index
    if args.chunk_size < len(missing_rels):
        chosen_rel_idxs = random.sample(missing_rels.tolist(), k=args.chunk_size)
    else:
        chosen_rel_idxs = missing_rels.tolist()
    existing_results.loc[chosen_rel_idxs,results_key] = 'R'
    with open(args.task_results_file, "w") as wf:
        existing_results.to_csv(wf, quoting=csv.QUOTE_NONE, sep='\t', header=True, index=False)
    chosen_rels = existing_results.loc[chosen_rel_idxs,'function'].tolist()
