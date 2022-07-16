import argparse
import copy
import numpy as np
import os
import random
from run_models.dataloaders import SynthDataLoader, SynthDataset
from run_models.pft_utils import set_extra_embeddings
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import GPT2Config, GPT2TokenizerFast, GPT2LMHeadModel
from transformers import T5Config, T5TokenizerFast, T5ForConditionalGeneration
from transformers import AdamW
import json
from run_models.utils import check_string_token_overlap
import math


parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='t5-base', choices=['t5-base', 'google/t5-v1_1-small', 'google/t5-v1_1-base', 'google/t5-v1_1-large', 'google/t5-v1_1-xl', 'google/t5-v1_1-xxl'])
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--compgen_eval', action='store_true', default=False, help='compositional generalization setup: train on atomic pieces, eval on composition')
parser.add_argument('--eval_batchsize', type=int, default=128)
parser.add_argument('--eval_interval', type=int, default=1, help="how often to evaluate (every <n> epochs)")
parser.add_argument('--eval_data', type=str, default='dev.jsonl', help="name of evaluation data under `args.data`")
parser.add_argument('--eval_only', action='store_true', default=False, help="only perform evaluation")
parser.add_argument('--data_formulation', type=str, default='generate', choices=['generate', 'generate_w_rel'])
parser.add_argument('--data', type=str, required=True, help="data directory under which `dev.jsonl` and `train.jsonl` reside")
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--force_save', action='store_true', default=False, help="overwrite checkpoint every epoch")
parser.add_argument('--test_memorization_only', action='store_true', default=False, help="only evaluate on training set")
parser.add_argument('--train_size', type=int, default=None, help="# of examples to train on (for few-shot learning)")
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
parser.add_argument('--valid_metric', type=str, default="accuracy", choices=["loss", "accuracy", "token_accuracy"], help="what metric to track during training. Specify `accuracy` for exact-match accuracy, `token_accuracy` for per-token accuracy (for sequences).")
parser.add_argument('--save_path', type=str, default=None, help="where to save/load checkpoints (automatically inferred if not specified)")
parser.add_argument('--seed', type=int, default=42, help="random seed")
parser.add_argument('--do_prompt_tune', action='store_true', default=False, help="flag for doing prompt tuning")
parser.add_argument('--n_prefix', type=int, default=100, help="number of prefix tokens to use for prompt tuning")
parser.add_argument('--no_pretrain', action='store_true', default=False, help="flag for using non-pretrained models")
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--use_segmentation_char', action='store_true', default=False, help="for sequences, whether to have a segmentation character in inputs and outputs")
args = parser.parse_args()

if args.compgen_eval: assert args.data_formulation == "generate_w_rel"
# args.workers = int((4 + torch.cuda.device_count() - 1) / torch.cuda.device_count())

# seed everything
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
if torch.cuda.device_count() > 0:
    torch.cuda.manual_seed_all(args.seed)
    print(f"Running on {torch.cuda.device_count()} GPUs", flush=True)
    # print(f"# of workers = {args.workers}", flush=True)

# get model
model_fp = args.arch
if 't5' in args.arch:
    model_class = T5ForConditionalGeneration
    config_class = T5Config
    tokenizer = T5TokenizerFast.from_pretrained('t5-base', local_files_only=True)
else:
    raise NotImplementedError

# load model/make save path
if not args.save_path:
    os.makedirs("checkpoints", exist_ok=True)
    gen_save_path = os.path.join(
        f"checkpoints",
        '/'.join(args.data.split('/')[1:])+("_compgen" if args.compgen_eval else "")+("_segment" if args.use_segmentation_char else ""),
        f"{'nopre_' if args.no_pretrain else ''}{args.arch}{args.data_formulation.replace('generate', '')}_lr{args.lr}_seed{args.seed}{f'_prompt{args.n_prefix}' if args.do_prompt_tune else ''}{f'_fs{args.train_size}' if args.train_size is not None else ''}.p",
    )
    mem_save_path = gen_save_path.replace('.p', '_mem.p')
elif '_mem.p' in args.save_path:
    mem_save_path = args.save_path
    gen_save_path = args.save_path.replace('_mem.p', '.p')
else:
    gen_save_path = args.save_path
    mem_save_path = gen_save_path.replace('.p', '_mem.p')
if len(os.path.split(gen_save_path)[0]) > 0:
    os.makedirs(os.path.split(gen_save_path)[0], exist_ok=True)
if len(os.path.split(mem_save_path)[0]) > 0:
    os.makedirs(os.path.split(mem_save_path)[0], exist_ok=True)

if args.test_memorization_only:
    # Load memorization checkpoint
    load_path = mem_save_path
else:
    # By default, loads generalization checkpoint
    # (if both exists)
    load_path = gen_save_path
if os.path.exists(load_path) or args.no_pretrain:
    print("Loading LM model")
    config = config_class.from_pretrained(model_fp, local_files_only=True)
    model = model_class(config)
else:
    print("Creating LM model")
    model = model_class.from_pretrained(model_fp, local_files_only=True)
print(f"    model save path: {load_path}")
model.config.max_length = tokenizer.model_max_length
# prompt tuning -- freeze all params except for additional tokens
if args.do_prompt_tune:
    # TODO
    for p in model.parameters():
        p.requires_grad = False
    set_extra_embeddings(args.arch, model, args.n_prefix)
if os.path.exists(load_path):
    model_dict = torch.load(load_path)
    model.load_state_dict(model_dict)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
if torch.cuda.is_available() > 0:
    model.to('cuda')
if args.eval_only:
    mem_save_file = mem_save_path.replace('.p', '.jsonl')
    gen_save_file = gen_save_path.replace('.p', '.jsonl')
    print(f"    pred save path: {mem_save_file} // {gen_save_file}")

# load optimizer
all_parameters = [p for p in model.parameters() if p.requires_grad]
print(f"# parameters to learn: {len(all_parameters)}")
optimizer = AdamW(all_parameters, lr=args.lr)

# load data
print(f"Tokenizer vocab size (after loading model): {len(tokenizer.vocab)}")
train_dataset = SynthDataset(
    os.path.join(args.data, 'train.jsonl'), data_formulation=args.data_formulation,
    tokenizer=tokenizer,
    do_prompt_tune=args.do_prompt_tune, n_prefix=args.n_prefix,
    data_size=args.train_size, randseed=args.seed,
    use_inner_tgts=args.compgen_eval, do_train=True,  # if compgen, train on inner functions
    use_segmentation_char=args.use_segmentation_char,
)
train_evalset = SynthDataset(
    os.path.join(args.data, 'train.jsonl'), data_formulation=args.data_formulation,
    tokenizer=tokenizer,
    do_prompt_tune=args.do_prompt_tune, n_prefix=args.n_prefix,
    data_size=args.train_size, randseed=args.seed,
    use_segmentation_char=args.use_segmentation_char,
)
# make reasonable size
test_memorization_only = args.test_memorization_only
if not args.test_memorization_only:
    dev_dataset = SynthDataset(
        os.path.join(args.data, f'{args.eval_data}'), data_formulation=args.data_formulation,
        tokenizer=tokenizer,
        do_prompt_tune=args.do_prompt_tune, n_prefix=args.n_prefix, randseed=args.seed,
        use_segmentation_char=args.use_segmentation_char,
    )
    test_memorization_only = test_memorization_only or len(dev_dataset) == 0
print(f"Tokenizer vocab size (after adding tokens): {len(tokenizer.vocab)}")
train_dataloader = SynthDataLoader(train_dataset, tokenizer, args.batchsize, shuffle=True, num_workers=0, pin_memory=True)
train_eval_dataloader = SynthDataLoader(train_evalset, tokenizer, args.eval_batchsize, num_workers=0, pin_memory=True)
if not test_memorization_only:
    dev_dataloader = SynthDataLoader(dev_dataset, tokenizer, args.eval_batchsize, num_workers=0, pin_memory=True,)


def evaluate(
    model, tokenizer, dataloader, save_path: str, prev_best_metrics: dict,
    metric: str = "accuracy", pred_save_file: str = None, save_checkpoint: bool = True,
    force_save: bool = False, use_segmentation_char: bool = False,
):
    model.eval()
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
    metrics = {
        'accuracy': 0,
        'loss': 0,
        'token_accuracy': 0,
    }
    max_pct_toks_correct = 0
    n_total = 0
    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if batch is None: continue
            bs = batch['context']['input_ids'].size(0)
            if torch.cuda.device_count() == 1:
                batch['context']['input_ids'] = batch['context']['input_ids'].to('cuda')
                batch['context']['attention_mask'] = batch['context']['attention_mask'].to('cuda')
                batch['tgt']['input_ids'] = batch['tgt']['input_ids'].to('cuda')
            outputs = model(
                input_ids=batch['context']['input_ids'], attention_mask=batch['context']['attention_mask'],
                labels=batch['tgt']['input_ids'], return_dict=True)
            preds = []
            gts = []
            if torch.cuda.device_count() > 1:
                generate_fn = model.module.generate
            else:
                generate_fn = model.generate
            if torch.cuda.is_available():
                batch['context_for_generate']['input_ids'] = batch['context_for_generate']['input_ids'].cuda()
                batch['context_for_generate']['attention_mask'] = batch['context_for_generate']['attention_mask'].cuda()
            generations = generate_fn(
                input_ids=batch['context_for_generate']['input_ids'],
                attention_mask=batch['context_for_generate']['attention_mask'],
            )
            for i, gt in enumerate(batch['tgt']['input_ids']):
                pred = tokenizer.decode(generations[i], skip_special_tokens=True)
                gt = tokenizer.decode(gt, skip_special_tokens=True)
                preds.append(pred)
                gts.append(gt)
                if dataloader.dataset.data_formulation.startswith("generate"):
                    if type(batch['all_valid_gens'][i]) == list:
                        # do sequence search
                        # candidate tokenizations
                        if use_segmentation_char:
                            pred_tokens = [tok.strip() for tok in pred.split('#')]
                            n_tokens_correct = 0
                            for tidx in range(min(len(pred_tokens), len(batch['all_valid_gens'][i]))):
                                n_tokens_correct += (pred_tokens[tidx] in batch['all_valid_gens'][i][tidx])
                            metrics['accuracy'] += n_tokens_correct == len(batch['all_valid_gens'][i])
                            metrics['token_accuracy'] += float(n_tokens_correct) / len(batch['all_valid_gens'][i])
                            best_pred_tok_split = pred_tokens
                        else:
                            max_n_toks_correct, best_pred_tok_split, best_valid_gen = check_string_token_overlap(
                                pred, batch['all_valid_gens'][i],
                                all_possible_words=batch['all_valid_gen_vocab'][i],
                                max_tok_len=batch['max_tok_len'][i],
                            )
                            metrics['accuracy'] += max_n_toks_correct == len(batch['all_valid_gens'][i])
                            metrics['token_accuracy'] += float(max_n_toks_correct) / len(batch['all_valid_gens'][i])
                    else:
                        assert type(batch['all_valid_gens'][i]) == set
                        if 'all_valid_gens' in batch:
                            metrics['accuracy'] += pred in batch['all_valid_gens'][i]
                        else:
                            metrics['accuracy'] += pred == gt
                else:
                    metrics['accuracy'] += pred == gt
            if pred_save_file is not None:
                # score all the possibilities
                if "all_valid_gen_tgts" in batch:
                    output_scores = []
                    for valid_gen_idx in range(0, batch['all_valid_gen_contexts']['input_ids'].size(0), dataloader.batch_size):
                        all_valid_gen_tgts = batch['all_valid_gen_tgts']['input_ids'][valid_gen_idx:valid_gen_idx+dataloader.batch_size].to('cuda')
                        valid_gen_batch_logits = model(
                            input_ids=batch['all_valid_gen_contexts']['input_ids'][valid_gen_idx:valid_gen_idx+dataloader.batch_size].to('cuda'),
                            attention_mask=batch['all_valid_gen_contexts']['attention_mask'][valid_gen_idx:valid_gen_idx+dataloader.batch_size].to('cuda'),
                            labels=all_valid_gen_tgts, return_dict=True,
                        ).logits
                        output_scores.append(loss_fct(valid_gen_batch_logits.permute(0,2,1), all_valid_gen_tgts).sum(1).to('cpu'))
                    output_scores = torch.cat(output_scores)
                for i, gt in enumerate(batch['tgt']['input_ids']):
                    context = tokenizer.decode(batch['context']['input_ids'][i], skip_special_tokens=True)
                    all_predictions.append({"prompt": context, "pred": preds[i], "gold": gts[i]})
                    if 'all_valid_gens' in batch:
                        if type(batch['all_valid_gens'][i]) == list:
                            all_predictions[-1]["all_valid_gens"] = [list(word_set) for word_set in batch['all_valid_gens'][i]]
                            all_predictions[-1]["pred_split"] = best_pred_tok_split
                        elif "all_valid_gen_tgts" in batch:
                            gen2score = {
                                tokenizer.decode(batch['all_valid_gen_tgts']['input_ids'][gen_idx], skip_special_tokens=True): output_scores[gen_idx].item()
                                for gen_idx in batch['all_valid_gen_indices'][i]
                            }
                            all_predictions[-1]["all_valid_gens"] = list(batch['all_valid_gens'][i])
                            all_predictions[-1]["all_valid_gen_scores"] = gen2score
            metrics['loss'] += outputs.loss.sum().item()
            n_total += bs
    metrics_stdout = []
    for m in metrics:
        metrics[m] /= n_total
        metrics_stdout.append(f'{m}: {metrics[m]}')
    print(f"EVAL METRICS - {', '.join(metrics_stdout)}, Total EXs: {n_total}")

    new_best_metrics = prev_best_metrics
    if save_checkpoint:
        if force_save or (metric == 'loss' and metrics[metric] < prev_best_metrics[metric]) or (
            metric != 'loss' and metrics[metric] > prev_best_metrics[metric]
        ) or (
            metric != 'loss' and metrics[metric] == prev_best_metrics[metric] and metrics['loss'] < prev_best_metrics['loss']
        ):
            print("NEW BEST MODEL")
            new_best_metrics = metrics
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
        else:
            print(f"eval {metric} went {'up' if metric == 'loss' else 'down'}")
    
    if pred_save_file is not None:
        with open(pred_save_file, 'w') as wf:
            for example in all_predictions:
                wf.write(json.dumps(example) + "\n")
    return metrics, new_best_metrics, int(n_total)

# initial eval
best_gen_epoch = 0
best_mem_epoch = 0
if args.valid_metric == 'accuracy' or args.valid_metric == 'token_accuracy':
    best_gen_metrics = {args.valid_metric: 0.0, 'loss': float('inf')}
    best_mem_metrics = {args.valid_metric: 0.0, 'loss': float('inf')}
elif args.valid_metric == 'loss':
    best_gen_metrics = {'loss': float('inf')}
    best_mem_metrics = {'loss': float('inf')}
else:
    raise NotImplementedError
print(f"INIT EVAL")
if not args.eval_only or test_memorization_only:
    # test mem
    print("MEMORIZATION")
    mem_metrics, best_mem_metrics, total = evaluate(
        model, tokenizer, train_eval_dataloader, mem_save_path, best_mem_metrics,
        metric=args.valid_metric, pred_save_file=mem_save_file if args.eval_only else None,
        save_checkpoint=False, use_segmentation_char=args.use_segmentation_char,
    )
if not test_memorization_only:
    # test gen
    print("GENERALIZATION")
    gen_metrics, best_gen_metrics, total = evaluate(
        model, tokenizer, dev_dataloader, gen_save_path, best_gen_metrics,
        metric=args.valid_metric, pred_save_file=gen_save_file if args.eval_only else None,
        save_checkpoint=False, use_segmentation_char=args.use_segmentation_char,
    )

if args.eval_only:
    exit()

model_init_dict = copy.deepcopy(model.state_dict())


# training loop
print("Start training")
for i in range(args.epochs):
    if (i - best_mem_epoch > args.patience) and (i - best_gen_epoch > args.patience): break
    model.train()
    for j, batch in enumerate(train_dataloader):
        if batch is None: continue
        optimizer.zero_grad()
        if torch.cuda.device_count() == 1:
            batch['context']['input_ids'] = batch['context']['input_ids'].to('cuda')
            batch['context']['attention_mask'] = batch['context']['attention_mask'].to('cuda')
            batch['tgt']['input_ids'] = batch['tgt']['input_ids'].to('cuda')
        outputs = model(
            input_ids=batch['context']['input_ids'], attention_mask=batch['context']['attention_mask'],
            labels=batch['tgt']['input_ids'], return_dict=True)
        loss = outputs.loss.mean()
        loss.backward()
        optimizer.step()
        if j%100 == 0:
            print(f"epoch {i}, batch {j}, loss: {loss.item()}", flush=True)
    new_dict = model.state_dict()
    # sanity check
    if args.do_prompt_tune:
        new_embed_weights = None
        for k in new_dict:
            if 'new_embed.weight' in k:
                assert new_embed_weights is None or (new_embed_weights == new_dict[k]).all()
                new_embed_weights = new_dict[k]
            else:
                assert (new_dict[k] == model_init_dict[k]).all()
    if i % args.eval_interval == -1 % args.eval_interval:
        print(f"EPOCH {i} EVAL")
        print("MEMORIZATION")
        mem_metrics, new_best_mem_metrics, total = evaluate(
            model, tokenizer, train_eval_dataloader, mem_save_path, best_mem_metrics,
            metric=args.valid_metric, force_save=args.force_save, use_segmentation_char=args.use_segmentation_char,
        )
        if new_best_mem_metrics != best_mem_metrics:
            # updated best model
            best_mem_epoch = i
            best_mem_metrics = new_best_mem_metrics
        new_best_gen_metrics = {args.valid_metric: 1.0}
        if not test_memorization_only:
            print("GENERALIZATION")
            gen_metrics, new_best_gen_metrics, total = evaluate(
                model, tokenizer, dev_dataloader, gen_save_path, best_gen_metrics,
                metric=args.valid_metric, force_save=args.force_save, use_segmentation_char=args.use_segmentation_char,
            )
            if new_best_gen_metrics != best_gen_metrics:
                # updated best model
                best_gen_epoch = i
                best_gen_metrics = new_best_gen_metrics
        torch.cuda.empty_cache()
        if math.isclose(new_best_gen_metrics[args.valid_metric], 1.0) and math.isclose(new_best_mem_metrics[args.valid_metric], 1.0):
            # early stopping
            print("Reached 100%% accuracy, stopping now.")
            break
