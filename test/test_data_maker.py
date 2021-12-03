import os
import json
from nltk.corpus import wordnet as wn
from glob import glob
from tqdm import tqdm
import itertools


data_dir = "wn_composites"
data_pattern = "wn_composites/*.jsonl"

FUNCTIONS = [
    'synonyms[eng]', 'synonyms[spa]',
    'antonyms[eng]', 'antonyms[spa]',
    'hypernyms[eng]', 'hypernyms[spa]',
    'hyponyms[eng]', 'hyponyms[spa]',
    'entailments[eng]', 'entailments[spa]',
    'translate[eng->spa]', 'translate[spa->eng]',
]

WORD2WNREL2GT = {}

def check_set_equal(set1, set2):
    if len(set(set1)) != len(set(set2)):
        return False
    set1 = set([word.replace('_', ' ') for word in set1])
    set2 = set([word.replace('_', ' ') for word in set2])
    return set1 == set2


def get_related_words(word: str, functions: list, lang_from: str, lang_to: str):
    related_words = {fxn: set() for fxn in functions}
    word_synsets = wn.synsets(word, lang=lang_from)
    for sidx, synset in enumerate(word_synsets):
        for fxn in related_words:
            if fxn.startswith('pos'):
                related_words[fxn].add(synset.pos())
        ss_in_relation_closure = {fxn: None for fxn in related_words}
        for fxn in related_words:
            if ss_in_relation_closure[fxn] is None:
                if fxn.startswith('hypernyms'):
                    ss_in_relation_closure[fxn] = set([i for i in synset.closure(lambda s: s.hypernyms())])
                if fxn.startswith('hyponyms'):
                    ss_in_relation_closure[fxn] = set([i for i in synset.closure(lambda s: s.hyponyms())])
                if fxn.startswith('entailments'):
                    ss_in_relation_closure[fxn] = set([i for i in synset.closure(lambda s: s.entailments())])
            if ss_in_relation_closure[fxn] is not None:
                for ss in ss_in_relation_closure[fxn]:
                    for l in ss.lemma_names(lang_to):
                        if l.lower() != word.lower():
                            related_words[fxn].add(l)
        for fxn in related_words:
            if fxn.startswith('synonyms') or fxn.startswith('translate'):
                for l in synset.lemmas(lang_to):
                    if l.name().lower() != word.lower():
                        related_words[fxn].add(l.name())
            if fxn.startswith('antonyms'):
                for l in synset.lemmas('eng'):
                    if l.antonyms():
                        for ant in l.antonyms():
                            ant_synset = ant.synset()  #wn.synsets(ant.name(), lang='eng')
                            # for ant_synset in ant_synsets:
                            for ant_synonyms in ant_synset.lemma_names(lang_to):
                                if ant_synonyms.lower() != word.lower():
                                    related_words[fxn].add(ant_synonyms)
    for fxn in related_words:
        WORD2WNREL2GT[word] = {fxn: related_words[fxn]}
    return related_words


def test_intersect_union(functions: list, wordlist: set, intersect_pred: dict = {}, union_pred: dict = {}):
    """
    intersection/union of functions in `functions`
    intersect_pred: word -> {words}
    union_pred: word -> {words}
    """
    lang_from = functions[0].split('[')[-1].split('>')[0]
    lang_to = functions[0].split('>')[-1].split(']')[0]

    for word in tqdm(wordlist):
        has_missing_function = False
        for fxn in functions:
            if word not in WORD2WNREL2GT or fxn not in WORD2WNREL2GT[word]:
                has_missing_function = True
                break
        
        if has_missing_function:
            related_words = get_related_words(word, functions, lang_from, lang_to)
        else:
            related_words = {fxn: WORD2WNREL2GT[word][fxn] for fxn in functions}

        function_intersection = None
        function_union = None
        for fxn in related_words:
            if intersect_pred is not None and function_intersection is None:
                function_intersection = related_words[fxn]
                function_union = related_words[fxn]
            else:
                function_intersection = function_intersection.intersection(related_words[fxn])
                function_union = function_union.union(related_words[fxn])
                
        assert intersect_pred.get(word, None) is None or check_set_equal(
            function_intersection, intersect_pred[word]), f"{word}: {function_intersection} != {intersect_pred[word]}"
        assert union_pred.get(word, None) is None or check_set_equal(
            function_union, union_pred[word]), f"{word}: {function_union} != {union_pred[word]}"


def test_nested(functions: list, wordlist: set, nested_pred: dict = {}):
    """
    `functions`: first function is outermost function, last is innermost function
    """
    for word in tqdm(wordlist):
        has_missing_function = False
        # keeps track of words at nth layer of functions
        related_words_currlayer = {word}
        related_words_nextlayer = set()
        for n in range(len(functions)-1,-1,-1):
            fxn = functions[n]
            lang_from = fxn.split('[')[-1].split('>')[0]
            lang_to = fxn.split('>')[-1].replace(']', '')
            for curr_word in related_words_currlayer:
                if curr_word not in WORD2WNREL2GT or fxn not in WORD2WNREL2GT[curr_word]:
                    curr_related_words = get_related_words(curr_word, [fxn], lang_from, lang_to)
                else:
                    curr_related_words = WORD2WNREL2GT[curr_word]
                related_words_nextlayer = related_words_nextlayer.union(curr_related_words[fxn])
            related_words_currlayer = list(related_words_nextlayer)
            related_words_nextlayer = set()

        assert nested_pred.get(word, None) is None or check_set_equal(
            related_words_currlayer, nested_pred[word]), f"{word}: {related_words_currlayer} != {nested_pred[word]}"


# test_nested(["synonyms[eng>eng]", "translate[spa>eng]"], wordlist=["comentarista"], nested_pred={"comentarista": ['percipient', 'reviewer', 'beholder', 'observer', 'commentator', 'perceiver']})
# test_nested(["antonyms[eng>eng]", "entailments[eng>eng]"], wordlist=["curved"], nested_pred={"curved": ['straighten', 'unbend', 'untwist']})
# test_nested(["antonyms[eng>eng]", "hypernyms[eng>eng]"], wordlist=["wrong"], nested_pred={"wrong": ["hm"]})
bad_functions = []
for f1, f2 in itertools.product(FUNCTIONS, repeat=2):
    """
    Intersection and Union
    """
    I_comp = f"intersection({f1}(0), {f2}(0))"
    I_file = os.path.join(data_dir, f'{I_comp}.jsonl')
    wordlist = set()
    I_pred = {}
    if os.path.exists(I_file):
        with open(I_file) as f:
            for line in f:
                line = json.loads(line)
                wordlist.add(line['inputs'][0])
                I_pred[line['inputs'][0]] = line['all_tgts']

    U_comp = f"union({f1}(0), {f2}(0))"
    U_file = os.path.join(data_dir, f'{U_comp}.jsonl')
    U_pred = {}
    if os.path.exists(U_file):
        with open(U_file) as f:
            for line in f:
                line = json.loads(line)
                wordlist.add(line['inputs'][0])
                U_pred[line['inputs'][0]] = line['all_tgts']

    if len(I_pred) > 0 or len(U_pred) > 0:
        print(f"Checking {I_comp} and {U_comp}")
        try:
            test_intersect_union([f1, f2], wordlist=wordlist, intersect_pred=I_pred, union_pred=U_pred)
        except AssertionError as e:
            print(e)
            print(I_comp)
            print(U_comp)
            bad_functions.append(I_comp)
            bad_functions.append(U_comp)

    """
    Composition
    """
    nested_comp = f"{f1}({f2}(0))"
    nested_file = os.path.join(data_dir, f'{nested_comp}.jsonl')
    wordlist = set()
    nested_pred = {}
    if os.path.exists(nested_file):
        # os.remove(nested_file)
        with open(nested_file) as f:
            for line in f:
                line = json.loads(line)
                wordlist.add(line['inputs'][0])
                nested_pred[line['inputs'][0]] = line['all_tgts']
    if len(nested_pred) > 0:
        print(f"Checking {nested_comp}")
        try:
            test_nested([f1, f2], wordlist=wordlist, nested_pred=nested_pred)
        except AssertionError as e:
            print(e)
            print(nested_comp)
            bad_functions.append(nested_comp)


print(bad_functions)