from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
import nltk
import json
import os
from function import Function, FUNCTION_REGISTRY, WordFunction
from random import Random


wordnet_pos = {
    'n': wn.NOUN,
    'v': wn.VERB,
    'a': wn.ADJ,
    'r': wn.ADV,
}

full_pos_name = {
    wn.NOUN: "nouns",
    wn.VERB: "verbs",
    wn.ADJ: "adjectives",
    wn.ADV: "adverbs",
}
full_lang_name = {
    "spa": "spanish",
    "eng": "english",
}

relation_nl_prefix = {
    "synonyms": "synonym of",
    "antonyms": "antonym of",
    "translate": "%BRACKET% translation of",
    "hypernyms": "hypernym of",
    "hyponyms": "hyponym of",
    "entailments": "entailment of",
    "POS": "part of speech of",
    "sentiment": "sentiment of",
    "is": "",
}

NL_ATTRIBUTE_NAMES = {
    'POS': '',
    'sentiment': 'sentiment'
}
NL_ATTRIBUTE_VALUES = {
    'POS': {
        'noun': 'noun',
        'verb': 'verb',
        'adj': 'adjective',
        'adv': 'adverb',
    },
    'sentiment': {
        'pos': 'positive',
        'neg': 'negative',
        'none': 'none or neutral',
    }
}
ATTR_VALUE_TO_WN_KEY = {
    'POS': {
        'noun': 'n',
        'verb': 'v',
        'adj': 'a',
        'adv': 'r',
    }
}

class GetWNRelatedWords(WordFunction):
    """
    Maps word set -> word set
    X -> {Y}
    Returns union over words with a particular relation w/ a set of words
    """
    def __init__(self, fn_tree, function, inner_fns, langs, attr_name=None, attr_value=None, **kwargs):
        super().__init__(fn_tree=fn_tree, inner_fns=inner_fns)

        # Getting pre-computed relations
        self.function = function
        # `langs` for *this* particular outer function only (not full function)
        self.input_lang = langs[0]
        self.output_lang = langs[1] if len(langs) > 1 else langs[0]
        self.lang = self.input_lang
        
        self.attr_name = attr_name
        self.attr_value = attr_value

        # input check
        if self.function == "is":
            assert self.attr_name is not None
            assert self.attr_value is not None
            if self.attr_name == "sentiment":
                assert self.input_lang == "eng"
            self.valid_set = self.get_valid_set(attr_name, attr_value)
        elif self.function == "translate":
            assert len(langs) == 2

        # cache, for this relation only (excluding inner functions)
        self.curr_relation_cache = {}

    @classmethod
    def get_func_name(cls):
        return [
            'antonyms', 'entailments', 'hypernyms', 'hyponyms', 'POS', 'sentiment', 'synonyms', 'translate',
        ]

    def to_nl(self):
        inner_nls = [inner_fn.to_nl() for inner_fn in self.inner_fns]
        assert len(inner_nls) == 1
        inner_nl = inner_nls[0]
        # predicates
        if self.function == "is":
            full_attr_name = NL_ATTRIBUTE_NAMES[self.attr_name]
            full_attr_value = NL_ATTRIBUTE_VALUES[self.attr_name][self.attr_value]
            return f"{inner_nl} is {full_attr_value} {full_attr_name}"

        # relations
        relation_nl = relation_nl_prefix[self.function]
        if self.function == "translate":
            relation_nl = relation_nl.replace("%BRACKET%", full_lang_name[self.output_lang])
        return f"{relation_nl} {inner_nl}".strip()
    
    def get_valid_set(self, attr_name, attr_value):
        valid_set = set()
        # for predicates, compute full set of inputs for which we return `True`
        if attr_name == "POS":
            # part-of-speech -> words
            all_pos_synsets = list(wn.all_synsets(ATTR_VALUE_TO_WN_KEY[attr_name][attr_value]))
            for sidx, synset in enumerate(all_pos_synsets):
                try:
                    for lemma in synset.lemma_names(self.output_lang):
                        valid_set.add(lemma)
                except nltk.corpus.reader.wordnet.WordNetError:
                    # lang not supported
                    continue
        elif attr_name == "sentiment":
            # sentiment -> words
            all_sent_synsets = list(swn.all_senti_synsets())
            for sidx, sent_synset in enumerate(all_sent_synsets):
                synset = sent_synset.synset
                if (
                    attr_value == 'pos' and sent_synset.pos_score() - sent_synset.neg_score() > 0
                ) or (
                    attr_value == 'neg' and sent_synset.pos_score() - sent_synset.neg_score() < 0
                ) or (
                    attr_value == 'none' and sent_synset.pos_score() - sent_synset.neg_score() == 0
                ):
                    for lemma in synset.lemma_names(self.output_lang):
                        valid_set.add(lemma)
        else:
            raise NotImplementedError
        return valid_set

    def __call__(self, inputs: list=None):
        """
        get all words lexically related to any words in word_set
        """
        all_outputs, missing_word_set = super().get_cached_words_results(inputs)
        missing_word_set = self.compute_inner_fns([missing_word_set])
        if len(missing_word_set) == 0:
            return {'out': all_outputs, 'inner': self.compute_inner_fns(inputs)}
        assert len(missing_word_set) == 1, f"Wordnet functions can only have 1 inner functions!: Currently {str(self)} has {len(missing_word_set)}"
        missing_word_set = missing_word_set[0]

        for word in missing_word_set:
            # iterate through words, computing appropriate relation
            if word in self.curr_relation_cache:
                all_outputs = all_outputs.union(self.curr_relation_cache[word])
                continue
            
            if self.function == "is":
                output = {word in self.valid_set}

            elif self.function == 'sentiment':
                output = set()
                # word -> sentiments
                word_sentiment_synsets = list(swn.senti_synsets(word))
                for sidx, synset in enumerate(word_sentiment_synsets):
                    if synset.pos_score() - synset.neg_score() > 0:
                        output.add('pos')
                    elif synset.pos_score() - synset.neg_score() < 0:
                        output.add('neg')
                    else:
                        output.add('none')

            else:
                output = set()
                # word -> related words
                word_synsets = wn.synsets(word, lang=self.input_lang)
                for sidx, synset in enumerate(word_synsets):
                    if self.function.startswith('pos'):
                        output.add(synset.pos())
                    ss_in_relation_closure = None
                    if self.function.startswith('hypernyms'):
                        ss_in_relation_closure = set([i for i in synset.closure(lambda s: s.hypernyms())])
                    if self.function.startswith('hyponyms'):
                        ss_in_relation_closure = set([i for i in synset.closure(lambda s: s.hyponyms())])
                    if self.function.startswith('entailments'):
                        ss_in_relation_closure = set([i for i in synset.closure(lambda s: s.entailments())])
                    if ss_in_relation_closure is not None:
                        for ss in ss_in_relation_closure:
                            for l in ss.lemma_names(self.output_lang):
                                if l.lower() != word.lower():
                                    output.add(l)
                    if self.function.startswith('synonyms') or self.function.startswith('translate'):
                        for l in synset.lemmas(self.output_lang):
                            if l.name().lower() != word.lower():
                                output.add(l.name())
                    if self.function.startswith('antonyms'):
                        for l in synset.lemmas('eng'):
                            if l.antonyms():
                                for ant in l.antonyms():
                                    ant_synset = ant.synset()
                                    for ant_synonyms in ant_synset.lemma_names(self.output_lang):
                                        if ant_synonyms.lower() != word.lower():
                                            output.add(ant_synonyms)
            self.curr_relation_cache[word] = output
            all_outputs = all_outputs.union(output)
        return {'out': all_outputs, 'inner': self.compute_inner_fns(inputs)}

    @classmethod
    def build(cls, fn_tree, inner_fns, **kwargs):
        curr_function = fn_tree.get_base_fn()
        function = str(fn_tree)
        # get langs
        assert len(fn_tree.bracket_children) > 0
        langs = str(fn_tree.bracket_children[0]).split('->')
        # get attributes (if `is`)
        attr_name, attr_value = None, None
        if curr_function == "is" and len(fn_tree.brace_children) > 0:
            assert len(fn_tree.brace_children) == 1
            attr_name = str(fn_tree.brace_children[0]).split('=')[0]
            attr_value = str(fn_tree.brace_children[0]).split('=')[1]

        return cls(
            fn_tree=fn_tree,
            function=curr_function,
            inner_fns=inner_fns,
            langs=langs,
            attr_name=attr_name,
            attr_value=attr_value,
            **kwargs,
        )
