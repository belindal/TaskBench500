from torch.utils.data import DataLoader, Dataset
import os
import json
import random
from tqdm import tqdm
from function.utils import convert_relation_to_nl
import torch
import itertools as it


def apply_mask_and_truncate(tensor, mask, max_len):
    """
    tensor (bsz, seqlen, *)
    mask (bsz)
    max_len (int)
    """
    return tensor[mask][:,:max_len]


class SynthDataset(Dataset):
    def __init__(
        self, fp, data_formulation: str, tokenizer = None,
        do_prompt_tune: bool = False, n_prefix: int = 0,
        data_size: int = None, randseed: int = 0, data: list=None,
        use_inner_tgts: bool=False, do_train: bool=False,
        use_segmentation_char: bool=False,
    ):
        """
        use_inner_tgts: set to train on inner functions (if more than 1, train on all of them, flattened)
        use_segmentation_char: use special character to demarcate token boundaries (for sequences)
        """
        self.fp = fp
        self.data_formulation = data_formulation

        self.tokenizer = tokenizer
        self.do_prompt_tune = do_prompt_tune  # prepend task tokens
        self.n_prefix = n_prefix
        self.data_size = data_size

        random.seed(randseed)

        if do_prompt_tune:
            task_tokens = ["<TASK{}>".format(str(i).zfill(2)) for i in range(self.n_prefix)]
            self.tokenizer.add_tokens(task_tokens)
            self.task_tokens = " ".join(task_tokens)

        self.use_inner_tgts = use_inner_tgts
        self.do_train = do_train
        self.use_segmentation_char = use_segmentation_char
        if data is not None:
            self.data = data
        else:
            self.data = self.load_data(fp)
    
    def copy(self):
        return SynthDataset(
            fp=self.fp, data_formulation=self.data_formulation,
            tokenizer=self.tokenizer,
            do_prompt_tune=self.do_prompt_tune, n_prefix=self.n_prefix,
            data_size=self.data_size, data=self.data.copy(),
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
    def load_data(self, fp):
        """
        `generate_input` and `input` are same except for in the case of using a GPT2 model,
        where `input` is the same as `output` (includes the label to compute loss),
        but we want to hide the label when generating
        """
        examples = []
        print("Loading data")
        # if there are lines w/ `Y2` and `Y`, prioritize them
        lines_w_both = []
        lines_w_one = []
        with open(fp) as f:
            for line in f:
                line = json.loads(line)
                if 'Y2' in line: lines_w_both.append(line)
                else: lines_w_one.append(line)
        function_name = os.path.split(os.path.split(fp)[0])[-1]
        random.shuffle(lines_w_both)
        random.shuffle(lines_w_one)
        for line in tqdm(it.chain(lines_w_both, lines_w_one)):
            if self.use_inner_tgts:
                line_outs = []
                for rel in line['comp_R']:
                    if len(line['comp_R'][rel]) == 0: continue
                    line_copy = line.copy()
                    line_copy['R'] = rel
                    # line_copy['train_tgts'] = [random.choice(line['comp_R'][rel])]
                    line_copy['all_tgts'] = line['comp_R'][rel]
                    line_outs.append(line_copy)
            else:
                line_outs = [line]
            for line_out in line_outs:
                if type(line_out['all_tgts'][0]) == list:
                    if self.use_segmentation_char:
                        join_char = " # "
                    else:
                        join_char = " "
                    line_inputs = [
                        join_char.join([word.lower() for word in inp])
                        if type(inp) == list else join_char.join([word.lower() for word in inp['ent_name']])
                        for inp in line_out['inputs']
                    ]
                    line_outputs = join_char.join([
                        random.choice(word_set).lower() if type(word_set[0]) == str else random.choice(word_set)['ent_name'].lower()
                        for word_set in line_out['all_tgts']
                    ])
                else:
                    line_inputs = [inp.lower() if type(inp) == str else inp['ent_name'].lower() for inp in line_out['inputs']]
                    line_outputs = line_out['all_tgts'][0].lower() if type(line_out['all_tgts'][0]) == str else line_out['all_tgts'][0]['ent_name'].lower()
                # skip negatives
                if self.data_formulation == "generate_w_rel":
                    if 'nl_R' not in line_out:
                        nl_relation = convert_relation_to_nl(line_out.get('R', function_name))
                    else:
                        nl_relation = line_out['nl_R']
                    for input_num in range(len(line_inputs)):
                        nl_relation = nl_relation.replace(f"%{input_num}%", line_inputs[input_num])
                    ex = {'input': f"{nl_relation} is ", 'output': line_outputs}
                else:
                    ex = {'input': f"{' | '.join(line_inputs)} | ", 'output': line_outputs}
                if 'all_tgts' in line_out:
                    if type(line_out['all_tgts'][0]) == list:
                        # sequences
                        if type(line_out['all_tgts'][0][0]) == str:
                            ex['all_valid_gens'] = [set(word_set) for word_set in line_out['all_tgts']]
                        else:
                            assert type(line_out['all_tgts'][0][0]) == dict
                            ex['all_valid_gens'] = [set([word['ent_name'].lower() for word in word_set]) for word_set in line_out['all_tgts']]
                        all_valid_gens_vocab = set().union(*ex['all_valid_gens'])
                        ex['all_valid_gen_vocab'] = set()
                        ex['max_tok_len'] = 1
                        # add prefixes
                        for word in all_valid_gens_vocab:
                            if word.count(' ') > 1:
                                word_toks = word.split(' ')
                                prefixes = set([' '.join(word_toks[:w]) for w in range(2,len(word_toks))])
                                ex['all_valid_gen_vocab'].union(prefixes)
                                ex['max_tok_len'] = max(ex['max_tok_len'], word.count(' ') + 1)
                            else:
                                ex['all_valid_gen_vocab'].add(word)
                    elif type(line_out['all_tgts'][0]) == dict:
                        ex['all_valid_gens'] = set([y['ent_name'].lower() for y in line_out['all_tgts']])
                    else:
                        ex['all_valid_gens'] = set([y.lower() for y in line_out['all_tgts']])
                if self.do_prompt_tune:
                    ex['input'] = f"{self.task_tokens} {ex['input']}"
                ex['generate_input'] = ex['input']
                examples.append(ex)
        if self.data_size is not None and len(examples) >= self.data_size:
            examples = examples[:self.data_size]
        random.shuffle(examples)
        return examples


class SynthDataLoader(DataLoader):
    def __init__(
        self, dataset, tokenizer, batch_size: int,
        shuffle: bool=True, num_workers=0, **kwargs
    ):
        super().__init__(dataset, batch_size, collate_fn=self.collate_fn, shuffle=shuffle, num_workers=num_workers, **kwargs)
        self.tokenizer = tokenizer
    
    def collate_fn(self, batch):
        new_batch = {}
        for i, item in enumerate(batch):
            for k in item:
                if k not in new_batch:
                    new_batch[k] = []
                new_batch[k].append(item[k])
        batch = new_batch
        context_tokens = self.tokenizer(batch['input'], return_tensors='pt', padding=True, truncation=False)
        # get contexts within max length of model
        items_to_keep = context_tokens['attention_mask'].sum(1) <= self.tokenizer.model_max_length
        if not items_to_keep.any():
            return None
        context_tokens = {k: apply_mask_and_truncate(context_tokens[k], items_to_keep, self.tokenizer.model_max_length) for k in context_tokens}

        tgt_tokens = self.tokenizer(batch['output'], return_tensors='pt', padding=True, truncation=False)
        tgt_tokens = {k: apply_mask_and_truncate(tgt_tokens[k], items_to_keep, self.tokenizer.model_max_length) for k in tgt_tokens}

        if 'generate_input' in batch:
            gen_input_tokens = self.tokenizer(batch['generate_input'], return_tensors='pt', padding=True, truncation=False)
            gen_input_tokens = {k: apply_mask_and_truncate(gen_input_tokens[k], items_to_keep, self.tokenizer.model_max_length) for k in gen_input_tokens}
            additional_inputs = {'context_for_generate': gen_input_tokens}
        if 'all_valid_gens' in batch:
            additional_inputs['all_valid_gens'] = batch['all_valid_gens']

            # tokenize
            if type(batch['all_valid_gens'][i]) != list:
                # store corresponding contexts for each gen
                # (bs * n_valid, seqlen)
                all_valid_gen_contexts_flat = {k: [] for k in context_tokens}
                # [bs * n_valid]
                all_valid_gens_flat = []
                context_idx2all_valid_idx = {}
                truncated_exidx = 0
                # create flattened contexts and valid_gens pairs
                # all_valid_gen_contexts_flat = [A,A,A,B,B,....]
                # all_valid_gens_flat         = [gen(A)_1, gen(A)_2, gen(A)_3, gen(B)_1, gen(B)_2, ...]
                # context_idx2all_valid_idx   = {0 -> [0,1,2], 1 -> [3,4]}
                for ex_idx, ex_valid_gens in enumerate(batch['all_valid_gens']):
                    if not items_to_keep[ex_idx]: continue
                    context_idx2all_valid_idx[truncated_exidx] = []
                    for gen in ex_valid_gens:
                        context_idx2all_valid_idx[truncated_exidx].append(len(all_valid_gens_flat))
                        for k in context_tokens: all_valid_gen_contexts_flat[k].append(context_tokens[k][truncated_exidx].to('cpu'))
                        all_valid_gens_flat.append(gen)
                    truncated_exidx += 1
                assert len(context_tokens['input_ids']) == truncated_exidx
                # print(all_valid_gens_flat)
                # tokenize each example
                flat_valid_gens_tokens = self.tokenizer(all_valid_gens_flat, return_tensors='pt', padding=True, truncation=False)
                for k in context_tokens: all_valid_gen_contexts_flat[k] = torch.stack(all_valid_gen_contexts_flat[k])
                
                # contexts and targets for all valid gens
                additional_inputs['all_valid_gen_contexts'] = all_valid_gen_contexts_flat
                additional_inputs['all_valid_gen_tgts'] = flat_valid_gens_tokens
                # indices of each example in all valid gens
                additional_inputs['all_valid_gen_indices'] = context_idx2all_valid_idx
        if 'all_valid_gen_vocab' in batch:
            additional_inputs['all_valid_gen_vocab'] = batch['all_valid_gen_vocab']
        if 'max_tok_len' in batch:
            additional_inputs['max_tok_len'] = batch['max_tok_len']

        return {'context': context_tokens, 'tgt': tgt_tokens, **additional_inputs}

