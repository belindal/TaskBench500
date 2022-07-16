MAX_TOKEN_LEN = 3

def check_string_token_overlap(pred: str, all_valid_gens: list, all_possible_words: list, max_tok_len: int):
    # find all ways to split `pred`
    pred_tok_cands = gen_all_possible_splits_of_len(
        pred, len(all_valid_gens), all_possible_words=all_possible_words, max_tok_len=max_tok_len,
    )
    # which way has max token overlap with gold? return that
    best_pred_tok_split = None
    max_n_toks_correct = -1
    best_valid_gen = None
    for pred_toks in pred_tok_cands:
        n_toks_correct = 0
        curr_matching_valid_gen = []
        for t, token in enumerate(pred_toks):
            n_toks_correct += token in all_valid_gens[t]
            curr_matching_valid_gen.append(token if token in all_valid_gens[t] else list(all_valid_gens[t])[0])
        if n_toks_correct > max_n_toks_correct:
            max_n_toks_correct = n_toks_correct
            best_pred_tok_split = pred_toks
            best_valid_gen = curr_matching_valid_gen
    return max_n_toks_correct, best_pred_tok_split, best_valid_gen


def gen_all_possible_splits_of_len(string, length, all_possible_words, max_tok_len=3):
    """
    Find all ways to split `string` up to a sequence length of `length`,
    where each word of string is in `all_possible_words` and is no longer than `max_tok_len`.
    """
    tokens = string.split(' ')
    if len(tokens) <= length:
        return [tokens]
    if len(tokens) > length * max_tok_len:
        tokens = tokens[:length * max_tok_len]
    token_len = len(tokens)
    all_splits = [tokens[:length]]
    all_splits_len = [tokens]
    while token_len > length:
        all_splits_len_new = []
        for split in all_splits_len:
            all_splits_len_new.extend(gen_all_possible_consecutive_merges(split, all_possible_words))
        if len(all_splits_len_new) == 0:
            break
        all_splits_len = all_splits_len_new
        all_splits.extend([tokens[:length] for tokens in all_splits_len])
        assert token_len - 1 == len(all_splits_len[0])
        token_len -= 1
    return all_splits

def gen_all_possible_consecutive_merges(tokens, all_possible_words):
    """
    generate all possible ways of merging consecutive elements of a list (`tokens`)
    where each merge must be in `all_possible_words`

    (`all_possible_words` contains vocab and all its prefixes)
    """
    all_possible = []
    for t in range(len(tokens)-1):
        # merge `t` and `t+1`
        curr_split = []
        if f"{tokens[t]} {tokens[t+1]}".count(' ') not in all_possible_words: continue
        for t1 in range(len(tokens)):
            if t1 == t:
                # ensure candidates aren't too long TODO set?
                curr_split.append(f"{tokens[t1]} {tokens[t1+1]}")
            elif t1 == t+1:
                continue
            else:
                curr_split.append(tokens[t1])
        all_possible.append(curr_split)
    return all_possible