from SPARQLWrapper import SPARQLWrapper, JSON
import os
import logging
from function import Function, FUNCTION_REGISTRY, WordFunction
import json
from function.utils import parse_to_function_tree
import pandas as pd


wdaccess_p = {
    'backend': "http://localhost:8890/sparql",
    'timeout': 10000,
    'global_result_limit': 10000,
    'logger': logging.getLogger(__name__),
    'use.cache': False,
    'mode': "quality"  # options: precision, fast
}

wiki_prop_save_fn = "resources/wikidata_property_ids.tsv"
wiki_entity_save_fn = "resources/wikidata_entity_ids.tsv"

def set_backend(backend_url):
    global sparql
    sparql = SPARQLWrapper(backend_url)
    sparql.setReturnFormat(JSON)
    sparql.setMethod("GET")
    sparql.setTimeout(wdaccess_p.get('timeout', 10000))
logger = wdaccess_p['logger']
logger.setLevel(logging.INFO)
sparql = None
set_backend(wdaccess_p.get('backend', "http://localhost:8890/sparql"))
GLOBAL_RESULT_LIMIT = wdaccess_p['global_result_limit']

FILTER_RELATION_CLASSES = "qr"

query_cache = {}
cached_counter = 0
query_counter = 1

cache_location = os.path.abspath(__file__)
cache_location = os.path.dirname(cache_location)
if wdaccess_p['use.cache'] and os.path.isfile(cache_location + "/.wdacess.cache"):
    try:
        with open(cache_location + "/.wdacess.cache") as f:
            query_cache = json.load(f)
        logger.info("Query cache loaded. Size: {}".format(len(query_cache)))
    except Exception as ex:
        logger.error("Query cache exists, but can't be loaded. {}".format(ex))
WIKIDATA_ENTITY_PREFIX = "http://www.wikidata.org/entity/"


def query_wikidata(query, prefix=WIKIDATA_ENTITY_PREFIX, use_cache=-1, timeout=-1):
    """
    Execute the following query against WikiData
    :param query: SPARQL query to execute
    :param prefix: if supplied, then each returned URI should have the given prefix. The prefix is stripped
    :param use_cache: set to 0 or 1 to override the global setting
    :param timeout: set to a value large than 0 to override the global setting
    :return: a list of dictionaries that represent the queried bindings
    """
    use_cache = (wdaccess_p['use.cache'] and use_cache != 0) or use_cache == 1
    global query_counter, cached_counter, query_cache
    query_counter += 1
    if use_cache and query in query_cache:
        cached_counter += 1
        return query_cache[query]
    if timeout > 0:
        sparql.setTimeout(timeout)
    sparql.setQuery(query)
    results = sparql.query().convert()
    # Change the timeout back to the default
    if timeout > 0:
        sparql.setTimeout(wdaccess_p.get('timeout', 10000))
    if "results" in results and len(results["results"]["bindings"]) > 0:
        results = results["results"]["bindings"]
        logger.debug("Results bindings: {}".format(results[0].keys()))
        if prefix:
            results = [r for r in results if all(not r[b]['value'].startswith("http://") or r[b]['value'].startswith(prefix) for b in r)]
        results = [{b: (r[b]['value'].replace(prefix, "") if prefix else r[b]['value']) for b in r} for r in results]
        if use_cache:
            query_cache[query] = results
        return results
    elif "boolean" in results:
        return results['boolean']
    else:
        logger.debug(results)
        return []

def get_wikidata_properties():
    if not os.path.exists(wiki_prop_save_fn):
        query = f"""
        SELECT DISTINCT ?prop ?propLabel
        WHERE {{
            ?prop rdf:type wikibase:Property .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """
        results_df = query_wikidata(query)
        results_df[['prop.value', 'propLabel.value']].to_csv(wiki_prop_save_fn, sep='\t')
    else:
        results_df = pd.read_csv(wiki_prop_save_fn, sep='\t')
    prop_name_to_id = {}
    for row in results_df[['prop.value', 'propLabel.value']].values:
        prop_name_to_id[row[1]] = os.path.split(row[0])[-1]
    return prop_name_to_id

def get_wikidata_entities():
    assert os.path.exists(wiki_entity_save_fn)
    entity_name_to_id = {}
    with open(wiki_entity_save_fn) as f:
        for line in f:
            line = line.strip().split('\t')
            entity_name_to_id[line[1].lower()] = line[0]
    return entity_name_to_id


class WikidataEntity:
    def __init__(self, Qid, ent_name):
        self.Qid = Qid.title()
        self.ent_name = ent_name
    
    def __str__(self):
        return self.ent_name
    
    def to_dict(self):
        return {"Qid": self.Qid, "ent_name": self.ent_name}
    
    @classmethod
    def from_dict(self, dict):
        return WikidataEntity(**json_dict)

    def getId(self):
        return self.Qid
    
    def has_name(self):
        return str(self.ent_name) != str(self.Qid) or not self.ent_name.startswith("Q")
    
    def __eq__(self, other):
        return self.Qid == other.Qid
    
    def __hash__(self):
        return hash(self.Qid)


relation_nl_prefix = {
    "mother": "mother of",
    "father": "father of",
    "mother[inv]": "child of",
    "father[inv]": "child of",
    "child": "child of",
    "child[inv]": "parent of",
    "location": "location of",
    "location[inv]": "something located at",
    "official language": "official language of",
    "creator": "creator of",
    "creator[inv]": "creation of",
    "occupation": "occupation of",
    "place of birth": "place of birth of",
    "date of birth": "date of birth of",
    "place of death": "place of death of",
    "employer": "employer of",
    "employer[inv]": "someone employed at",
    "head of state": "head of state of",
    "head of state[inv]": "state headed by", #???
    "position held": "position held by",
    "sex or gender": "sex or gender of",
    "developer": "developer of",
    "performer": "performer of",
    "owned by": "owner of",
    "named after": "namesake of",
    "influenced by": "influence of",
    "country of citizenship": "country of citizenship of",
    "member of sports team": "sports team of",
    "member of political party": "political party of",
    "languages spoken written or signed": "language of",
    "record label": "record label of",
    "native language": "native language of",
    "genre": "genre of",
    "subclass of": "subclass of",
    "official language": "official language of",
    "position played on team": "position played by",
    "original language of film or TV show": "original language of",
    "has part": "part of",
    "diplomatic relation": "country with diplomatic relation to",
    "manufacturer": "manufacturer of",
    "continent": "continent of",
    "country": "country of",
    "country of origin": "country of origin of",
    "instance of": "type of",
}

class GetWikiRelatedWords(WordFunction):
    """
    wiki[condition](X)
    """
    rel_name2id: dict = get_wikidata_properties()
    ent_name2id: dict = get_wikidata_entities()
    def __init__(
        self, fn_tree, inner_fns,
        condition_fntree, max_results: int = 10000, suppress_print=False, **kwargs,
    ):
        super().__init__(fn_tree=fn_tree, inner_fns=inner_fns)
        self.condition_fntree = condition_fntree
        self.is_predicate = condition_fntree.get_base_fn() in ["is", "has", "land", "lor"]
        self.max_results = max_results
        self.all_samples = None
        self.suppress_print = suppress_print

        if self.is_predicate:
            condition_pos, var2fxn_pos = convert_fntree2query(condition_fntree, query_options_flag='sample_pos')
            condition_neg, var2fxn_neg = convert_fntree2query(condition_fntree, query_options_flag='sample_neg')
            sample_query_pos, ordered_input_variables_pos = self.make_sparql_query(condition_pos, self.max_results, get_inners=False)
            sample_query_neg, ordered_input_variables_neg = self.make_sparql_query(condition_neg, self.max_results, get_inners=False)
            self.pos_sample_query_infos = {"query": sample_query_pos, "input_vars": ordered_input_variables_pos, "var2fxn": var2fxn_pos}
            self.neg_sample_query_infos = {"query": sample_query_neg, "input_vars": ordered_input_variables_neg, "var2fxn": var2fxn_neg}
        else:
            sample_condition, sample_var2fxn = convert_fntree2query(condition_fntree, query_options_flag='sample')
            sample_query, sample_ordered_input_variables = self.make_sparql_query(sample_condition, self.max_results, get_inners=False)
            self.sample_query_infos = {"query": sample_query, "input_vars": sample_ordered_input_variables, "var2fxn": sample_var2fxn}
        input_condition, input_var2fxn = convert_fntree2query(condition_fntree, query_options_flag='input')
        input_query, inputq_ordered_input_variables = self.make_sparql_query(input_condition, self.max_results, get_inners=False)
        self.input_query_infos = {"query": input_query, "input_vars": inputq_ordered_input_variables, "var2fxn": input_var2fxn}

        self.inner_conditions = {str(child): convert_fntree2query(child, query_options_flag='input')[0] for child in condition_fntree.paren_children if len(child.paren_children) > 0}
        self.inner_queries = {
            fxn: self.make_sparql_query(self.inner_conditions[fxn], self.max_results, get_inners=False)[0]
            for fxn in self.inner_conditions
        }

    @classmethod
    def get_func_name(cls):
        return ["wiki"]
    
    def to_nl(self):
        return self._to_nl(self.condition_fntree)
    
    def _to_nl(self, tree):
        # TODO order of operations???
        if len(tree.paren_children) == 0:
            return f"%{str(tree)}%"
        else:
            inner_nls = [self._to_nl(subtree) for subtree in tree.paren_children]
            condition_nl = tree.get_base_fn()
            if condition_nl in relation_nl_prefix:
                assert len(inner_nls) == 1
                if len(tree.bracket_children) > 0 and str(tree.bracket_children[0]) == "inv":
                    condition_nl = f"{condition_nl}[inv]"
                relation_nl = relation_nl_prefix[condition_nl]
                return f"{relation_nl} {inner_nls[0]}".strip()
            elif condition_nl in ["union", "intersection"]:
                assert len(inner_nls) == 2
                return f"{condition_nl} of {inner_nls[0]} and {inner_nls[1]}"
            elif condition_nl in ["lor", "land"]:
                assert len(inner_nls) == 2
                return f"{inner_nls[0]} {condition_nl[1:]} {inner_nls[1]}"
            elif condition_nl in ["lnot"]:
                assert len(inner_nls) == 1
                return f"{condition_nl[1:]} {inner_nls[0]}"
            elif condition_nl in ["is", "has"]:
                assert len(inner_nls) == 1
                predicate = str(tree.brace_children[0]).split('=')
                return f"{relation_nl_prefix[predicate[0]]} {inner_nls[0]} {condition_nl} {predicate[1]}"
            else:
                raise NotImplementedError

    def make_sparql_query(self, condition, max_n_results, get_inners=False):
        """
        get_inners: get outputs of inner functions
        """
        # TODO add langs other than english??
        variables = set()
        condition_toks = condition.split()
        for t, term in enumerate(condition_toks):
            if term.startswith("?"): #and ("inp" in term or term == "?out"):
                variables.add(term)

        select_clause = []
        lang_filter_clauses = []  # filter lang for each variable
        group_by_inputs_clause = []  # group by all inputs
        ordered_input_variables = []
        assert "?out" in variables
        for var in variables:
            if var == "?out":
                if self.is_predicate:
                    select_clause.append("?out")
                else:
                    select_clause.append(f"(GROUP_CONCAT(?out; separator='|') as ?outs)")
                    select_clause.append(f"(GROUP_CONCAT(?outLabel; separator='|') as ?outLabels)")
            elif var.startswith("?inp"):
                select_clause.append(var)
                select_clause.append(f"{var}Label")
                if not self.is_predicate:
                    group_by_inputs_clause.append(var)
                    group_by_inputs_clause.append(f"{var}Label")
                ordered_input_variables.append(var.replace('?', ''))
            elif get_inners:
                # add intermediate variables
                if self.is_predicate:
                    select_clause.append(f"?{var}")
                else:
                    select_clause.append(f"(GROUP_CONCAT({var}; separator='|') as {var}s)")
                    select_clause.append(f"(GROUP_CONCAT({var}Label; separator='|') as {var}Labels)")
            if (not self.is_predicate and var == "?out") or var.startswith("?inp") or get_inners:
                lang_filter_clauses.append(
                    f'{{ GRAPH <http://wikidata.org/terms> {{ {var} rdfs:label {var}Label }} FILTER ( lang({var}Label) = "en" ) }}'
                )
        ordered_input_variables = sorted(ordered_input_variables)
        # ?inp ?inpLabel (GROUP_CONCAT(?out; separator='|') as ?outs) (GROUP_CONCAT(?outLabel; separator='|') as ?outLabels)
        select_clause = " ".join(select_clause)
        # ?inp ?inpLabel
        group_by_inputs_clause = " ".join(group_by_inputs_clause)
        # { GRAPH <http://wikidata.org/terms> { ?inp rdfs:label ?inpLabel } FILTER ( lang(?inpLabel) = "en" ) }
        # { GRAPH <http://wikidata.org/terms> { ?out rdfs:label ?outLabel } FILTER ( lang(?outLabel) = "en" ) }
        lang_filter_clauses = "\n".join(lang_filter_clauses)

        query = f"""
PREFIX g:<http://wikidata.org/>
PREFIX e:<http://www.wikidata.org/entity/>
PREFIX rdfs:<http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos:<http://www.w3.org/2004/02/skos/core#>
PREFIX base:<http://www.wikidata.org/ontology#>
PREFIX schema:<http://schema.org/>

SELECT {select_clause}
WHERE {{
    {{ GRAPH <http://wikidata.org/statements> {{
{condition}
    }} }}
    {lang_filter_clauses}
}}
{"GROUP BY" if len(group_by_inputs_clause) > 0 else ""} {group_by_inputs_clause}
LIMIT {max_n_results}
        """
        # TODO add order here?
        if not self.suppress_print:
            print(query)
        return query, ordered_input_variables

    def __call__(self, inputs: list=None):
        # TODO `inner_fns`?
        # related_ents, missing_ent_set = super().get_cached_words_results(inputs)
        assert len(inputs) == 1 and len(inputs[0]) == 1
        input_ents = list(inputs[0])[0]
        # if self.all_samples is None:
        #     self.get_samples()
        # if False:  #self.all_samples is not None:
        #     # inner_results = [query(ent) for query in self.inner_queries]
        #     input_ent_tuple = tuple([input_ent.getId() for input_ent in inputs])
        #     if input_ents in self.all_samples:
        #         return self.all_samples[input_ents]
        # TODO
        input_query = self.input_query_infos['query']
        inner_queries = self.inner_queries.copy()
        for entidx, input_ent in enumerate(input_ents):
            input_query = input_query.replace(f"%INPUT{entidx}%", f"e:{input_ent.getId()}")
            inner_queries = {fxn: inner_queries[fxn].replace(f"%INPUT{entidx}%", f"e:{input_ent.getId()}") for fxn in inner_queries}
        query_results = self.get_query_results(input_query, [], self.input_query_infos['var2fxn'])[()]['out']
        results = {
            "out": self.get_query_results(input_query, [], self.input_query_infos['var2fxn'])[()]['out'],
            "inner": {fxn: self.get_query_results(inner_queries[fxn], [])[()]['out'] for fxn in inner_queries},
        }
        # filter inner
        for fxn in inner_queries:
            for i, inner_fxn_result in enumerate(results['inner'][fxn]):
                if type(inner_fxn_result) == WikidataEntity and len(inner_fxn_result.getId()) == 0:
                    del results['inner'][fxn][i]
            if len(results['inner'][fxn]) == 0:
                del results['inner'][fxn]
        return results

    def get_samples(self):
        # get samples
        if self.all_samples is not None:  #and (not self.is_predicate and self.sample_query == self.input_query):
            return set(self.all_samples.keys())
        if self.is_predicate:
            wiki_results_dict_pos = self.get_query_results(self.pos_sample_query_infos['query'], self.pos_sample_query_infos['input_vars'], self.pos_sample_query_infos['var2fxn'])
            wiki_results_dict_neg = self.get_query_results(self.neg_sample_query_infos['query'], self.neg_sample_query_infos['input_vars'], self.neg_sample_query_infos['var2fxn'])
            wiki_results_dict = {**wiki_results_dict_pos, **wiki_results_dict_neg}
        else:
            wiki_results_dict = self.get_query_results(self.sample_query_infos['query'], self.sample_query_infos['input_vars'], self.sample_query_infos['var2fxn'])
        # if self.sample_queries != self.input_queries:
        #     return wiki_results_dict.keys()
        self.all_samples = wiki_results_dict
        return set(self.all_samples.keys())
    
    def get_query_results(self, query, input_vars, var2fxn=None):
        # get wiki results and format
        raw_wiki_results = query_wikidata(query)
        wiki_results_dict = {}
        for sample in raw_wiki_results:
            inputs = [WikidataEntity(Qid=sample[inp_var], ent_name=sample[f'{inp_var}Label']) for inp_var in input_vars]
            if self.is_predicate:
                wiki_results_dict[tuple(inputs)] = {
                    'out': [sample['out']],
                }
            else:
                wiki_results_dict[tuple(inputs)] = {'out': [
                    WikidataEntity(Qid=outid, ent_name=sample['outLabels'].split('|')[o])
                    for o, outid in enumerate(sample['outs'].split('|'))
                ]}
        return wiki_results_dict

    @classmethod
    def build(cls, fn_tree, inner_fns, max_results: int=10000, **kwargs):
        # TODO
        relation = fn_tree.get_base_fn()
        condition_fntree = fn_tree.brace_children[0]
        return cls(
            fn_tree=fn_tree,
            inner_fns=inner_fns,
            condition_fntree=condition_fntree,
            max_results=max_results,
            **kwargs,
        )

def query_of_relation(relation_id: str, max_n_results: int = None):
    # e1 <REL> e2
    query = f"""
    PREFIX g:<http://wikidata.org/>
    PREFIX e:<http://www.wikidata.org/entity/>
    PREFIX rdfs:<http://www.w3.org/2000/01/rdf-schema#>
    PREFIX skos:<http://www.w3.org/2004/02/skos/core#>
    PREFIX base:<http://www.wikidata.org/ontology#>
    PREFIX schema:<http://schema.org/>

    SELECT ?inp0 ?inp0Label (GROUP_CONCAT(?out; separator='|') as ?outs) (GROUP_CONCAT(?outLabel; separator='|') as ?outLabels)
    WHERE
    {{
    {{
    {{GRAPH <http://wikidata.org/statements> {{ ?inp0 e:%RELID%s/e:%RELID%v ?out . }}}}
    }}

    {{
    GRAPH <http://wikidata.org/terms> {{ ?inp0 rdfs:label ?inp0Label }}
    FILTER ( lang(?inp0Label) = "en" )
    }}
    {{
    GRAPH <http://wikidata.org/terms> {{ ?out rdfs:label ?outLabels }}
    FILTER ( lang(?outLabels) = "en" )
    }}
    }}
    GROUP BY ?inp0 ?inp0Label
    LIMIT %MAXRESULTS%
    """
    # (group_concat(?label1;separator="/") as ?label1s) (group_concat(?e2;separator="/") as ?entity2s) (group_concat(?label2;separator="/") as ?label2s)
    query = query.replace("%RELID%", relation_id)
    if max_n_results is None:
        query = query.replace(" LIMIT %MAXRESULTS%", "")
    else:
        query = query.replace("%MAXRESULTS%", str(max_n_results))
    # {{GRAPH <http://wikidata.org/statements> {{ ?e1 e:P22s ?m . ?m ?p ?e2 . }}}}
    # {{GRAPH <http://wikidata.org/statements> {{ ?e1 ?p ?m . ?m e:P22v ?e2 . }}}}
    # {{GRAPH <http://wikidata.org/statements> {{ ?e1 e:P22s/e:P22v ?e2 . }}}}
    # {{GRAPH <http://wikidata.org/statements> {{ ?e1 e:P22s/e:P22v e:Q42 . }}}}
    print(query)
    results = query_wikidata(query)
    return results


def convert_fntree2query(fn_tree, query_options_flag="input"):#, get_inners:bool=False):
    """
    convert from function parentheses format to sparql query format
    `query_options_flag`: this flag specifies which query to get
        Options: For predicates, can be `sample_pos`/`sample_neg`/`input`. For relations, can be `sample`/`input`.
    """
    query, out_var, var2fxn = _convert_fntree2query(fn_tree, var2fxn=[], query_options_flag=query_options_flag)
    query = query.replace(out_var, "?out")
    try:
        var2fxn[int(out_var.replace("?v", ""))] = str(fn_tree)
    except:
        import pdb; pdb.set_trace()
    return query, var2fxn


def _convert_fntree2query(fn_tree, var2fxn: list=[], query_options_flag="input"):#, get_inners:bool=False):
    """
    convert from our function format to sparql queries (equally sample from each query)
    var2fxn used to indicate existing intermediate variables and what they represent

    (only ever returns list of queries > 1 in `is` or `has` case, which is an output case)
    `query_options_flag`: this flag specifies which query to get
        Options: For predicates, can be `sample_pos`/`sample_neg`/`input`. For relations, can be `sample`/`input`.
    """
    # returns query_clause, out_var
    # TODO FIX!!!
    # Write everything out explicitly as sentences, wth subjects...
    if len(fn_tree.paren_children) == 0:
        return "", f"%INPUT{str(fn_tree)}%" if query_options_flag == 'input' else f"?inp{str(fn_tree)}", var2fxn
    else:
        if fn_tree.get_base_fn() == "is":
            assert len(fn_tree.brace_children) == 1
            rel_name_and_obj = fn_tree.brace_children[0].get_base_fn().split('=')
            rel_name, rel_obj = rel_name_and_obj[0], rel_name_and_obj[1]
            rel_id = GetWikiRelatedWords.rel_name2id[rel_name]
            assert len(fn_tree.paren_children) == 1
            # get inner query
            inner_query, inner_out_var, var2fxn = _convert_fntree2query(fn_tree.paren_children[0], var2fxn=var2fxn, query_options_flag=query_options_flag)
            obj_id = GetWikiRelatedWords.ent_name2id[rel_obj]
            new_var = f"?v{len(var2fxn)}"
            var2fxn.append(str(fn_tree))
            if query_options_flag == 'sample_pos':
                condition_clause = f"{inner_out_var} e:{rel_id}s/e:{rel_id}v e:{obj_id} . BIND( true as {new_var} ) . {inner_query}"
            elif query_options_flag == 'sample_neg':
                condition_clause = f"FILTER( ?tmp{len(var2fxn)} != e:{obj_id} ) . {inner_out_var} e:{rel_id}s/e:{rel_id}v ?tmp{len(var2fxn)} . BIND( false as {new_var} ) . {inner_query}"
            else:
                assert query_options_flag == 'input'
                condition_clause = f"{inner_out_var} e:{rel_id}s/e:{rel_id}v ?tmp{len(var2fxn)} . BIND( ?tmp{len(var2fxn)} = e:{obj_id} as {new_var} ) . {inner_query}"
            return condition_clause, new_var, var2fxn
        elif fn_tree.get_base_fn() == "land":
            assert len(fn_tree.paren_children) == 2
            if query_options_flag == 'sample_pos' or query_options_flag == 'input':
                inner_query_clause0, inner_out_var0, var2fxn = _convert_fntree2query(fn_tree.paren_children[0], var2fxn=var2fxn, query_options_flag=query_options_flag)
                inner_query_clause1, inner_out_var1, var2fxn = _convert_fntree2query(fn_tree.paren_children[1], var2fxn=var2fxn, query_options_flag=query_options_flag)
                new_var = f"?v{len(var2fxn)}"
                var2fxn.append(str(fn_tree))
                condition_clause = f"{inner_query_clause0} {inner_query_clause1}\nBIND( {inner_out_var0} && {inner_out_var1} as {new_var} ) ."
            else:
                assert query_options_flag == 'sample_neg'
                inner_query_clause0, inner_out_var0, var2fxn = _convert_fntree2query(fn_tree.paren_children[0], var2fxn=var2fxn, query_options_flag=query_options_flag)
                inner_query_clause1, inner_out_var1, var2fxn = _convert_fntree2query(fn_tree.paren_children[1], var2fxn=var2fxn, query_options_flag=query_options_flag)
                new_var = f"?v{len(var2fxn)}"
                var2fxn.append(str(fn_tree))
                clause0_parts = inner_query_clause0.split('.')
                clause1_parts = inner_query_clause1.split('.')
                # combine filter clauses
                assert clause0_parts[0].startswith("FILTER")
                assert clause1_parts[0].startswith("FILTER")
                filter_clause0 = clause0_parts[0].strip().replace("FILTER(", "")[:-1].strip()
                filter_clause1 = clause1_parts[0].strip().replace("FILTER(", "")[:-1].strip()
                inner_query_clause0 = '.'.join(clause0_parts[1:]).strip()
                inner_query_clause1 = '.'.join(clause1_parts[1:]).strip()
                # either condition violated
                condition_clause = f"FILTER( {filter_clause0} || {filter_clause1} ) . {inner_query_clause0} {inner_query_clause1}\nBIND( {inner_out_var0} && {inner_out_var1} as {new_var} ) ."
            return condition_clause, new_var, var2fxn
        elif fn_tree.get_base_fn() == "lor":
            assert len(fn_tree.paren_children) == 2
            if query_options_flag == 'input' or query_options_flag == 'sample_neg':
                inner_query_clause0, inner_out_var0, var2fxn = _convert_fntree2query(fn_tree.paren_children[0], var2fxn=var2fxn, query_options_flag=query_options_flag)
                inner_query_clause1, inner_out_var1, var2fxn = _convert_fntree2query(fn_tree.paren_children[1], var2fxn=var2fxn, query_options_flag=query_options_flag)
                new_var = f"?v{len(var2fxn)}"
                var2fxn.append(str(fn_tree))
                condition_clause = f"{inner_query_clause0} {inner_query_clause1}\nBIND( {inner_out_var0} || {inner_out_var1} as {new_var} ) ."
            else:
                assert query_options_flag == 'sample_pos'
                inner_query_clause0, inner_out_var0, var2fxn = _convert_fntree2query(fn_tree.paren_children[0], var2fxn=var2fxn, query_options_flag=query_options_flag)
                inner_query_clause1, inner_out_var1, var2fxn = _convert_fntree2query(fn_tree.paren_children[1], var2fxn=var2fxn, query_options_flag=query_options_flag)
                new_var = f"?v{len(var2fxn)}"
                var2fxn.append(str(fn_tree))
                clause0_parts = inner_query_clause0.split('.')
                clause1_parts = inner_query_clause1.split('.')
                # replace ents with vars; add filter for either first ent or second ent
                assert len(clause0_parts[0].strip().split()) == 3
                assert len(clause1_parts[0].strip().split()) == 3
                tgt_ent0 = clause0_parts[0].strip().split()[-1]
                tgt_ent1 = clause1_parts[0].strip().split()[-1]
                inner_query_clause0 = inner_query_clause0.replace(tgt_ent0, f"?tmp{len(var2fxn)}")
                inner_query_clause1 = inner_query_clause1.replace(tgt_ent1, f"?tmp{len(var2fxn)}_1")
                condition_clause = f"FILTER( ?tmp{len(var2fxn)} = {tgt_ent0} || ?tmp{len(var2fxn)}_1 = {tgt_ent1} ) . {inner_query_clause0} {inner_query_clause1}\nBIND( {inner_out_var0} || {inner_out_var1} as {new_var} ) ."
            return condition_clause, new_var, var2fxn
        elif fn_tree.get_base_fn() == "lnot":
            assert len(fn_tree.paren_children) == 1
            inner_query_clause0, inner_out_var0, var2fxn = _convert_fntree2query(fn_tree.paren_children[0], var2fxn=var2fxn, query_options_flag=query_options_flag)
            new_var = f"?v{len(var2fxn)}"
            var2fxn.append(str(fn_tree))
            condition_clause = f"BIND( !( {inner_out_var0} ) as {new_var} ) .\n{inner_query_clause0} .\n{inner_query_clause1}"
            return condition_clause, new_var, var2fxn
        elif fn_tree.get_base_fn() == "intersection":
            assert len(fn_tree.paren_children) == 2
            inner_query_clause0, inner_out_var0, var2fxn = _convert_fntree2query(fn_tree.paren_children[0], var2fxn=var2fxn, query_options_flag=query_options_flag)
            inner_query_clause1, inner_out_var1, var2fxn = _convert_fntree2query(fn_tree.paren_children[1], var2fxn=var2fxn, query_options_flag=query_options_flag)
            # if get_inners:
            #     # TODO intermediate variables
            #     import pdb; pdb.set_trace()
            #     new_var = f"?v{len(var2fxn)}"
            #     var2fxn.append(str(fn_tree))
            # else:
            # don't change output variable
            new_var = inner_out_var0
            inner_query_clause1 = inner_query_clause1.replace(inner_out_var1, inner_out_var0)
            return f"{inner_query_clause0} {inner_query_clause1}", new_var, var2fxn
        elif fn_tree.get_base_fn() == "union":
            assert len(fn_tree.paren_children) == 2
            inner_query_clause0, inner_out_var0, var2fxn = _convert_fntree2query(fn_tree.paren_children[0], var2fxn=var2fxn, query_options_flag=query_options_flag)
            inner_query_clause1, inner_out_var1, var2fxn = _convert_fntree2query(fn_tree.paren_children[1], var2fxn=var2fxn, query_options_flag=query_options_flag)
            # change output variable
            new_var = f"?v{len(var2fxn)}"
            condition_clause = f"{{ {inner_query_clause0.replace(inner_out_var0, new_var)} }}\nUNION\n{{ {inner_query_clause1.replace(inner_out_var1, new_var)} }}\n{inner_query_clause1} {inner_query_clause0}"
            # add new var to map
            var2fxn.append(str(fn_tree))
            """
            # if get_inners:
            #     new_var = f"?v{len(var2fxn)}"
            #     condition_clause = f"{{ {inner_query_clause0} OPTIONAL {{ {inner_query_clause1.replace(inner_out_var1, new_var)} }} }}\nUNION\n{{ {inner_query_clause1} OPTIONAL {{ {inner_query_clause0.replace(inner_out_var0, new_var)} }} }}"
            #     var2fxn.append(str(fn_tree))
            # else:
            new_var = inner_out_var0
            # condition_clause = f"{{ {inner_query_clause0} OPTIONAL {{ {inner_query_clause1.replace(inner_out_var1, new_var)} }} }}\nUNION\n{{ {inner_query_clause1.replace(inner_out_var1, new_var)} OPTIONAL {{ {inner_query_clause0} }} }}"
            condition_clause = f"{{ {inner_query_clause0} }}\nUNION\n{{ {inner_query_clause1.replace(inner_out_var1, new_var)} }}"
            # """
            return condition_clause, new_var, var2fxn
        elif fn_tree.get_base_fn() in GetWikiRelatedWords.rel_name2id:
            rel_id = GetWikiRelatedWords.rel_name2id[fn_tree.get_base_fn()]
            assert len(fn_tree.paren_children) == 1
            inner_query, inner_out_var, var2fxn = _convert_fntree2query(fn_tree.paren_children[0], var2fxn=var2fxn, query_options_flag=query_options_flag)
            # change output variable
            new_var = f"?v{len(var2fxn)}"
            if len(fn_tree.bracket_children) > 0 and str(fn_tree.bracket_children[0]) == "inv":
                # invert relation
                query = f"{new_var} e:{rel_id}s/e:{rel_id}v {inner_out_var} . {inner_query}"
            else:
                query = f"{inner_out_var} e:{rel_id}s/e:{rel_id}v {new_var} . {inner_query}"
            # add new var to map
            var2fxn.append(str(fn_tree))
            return query, new_var, var2fxn
        else:
            raise NotImplementedError


def test_wikidata_query():
    max_n_results = 2000
    results = query_wikidata("""
PREFIX g:<http://wikidata.org/>
PREFIX e:<http://www.wikidata.org/entity/>
PREFIX rdfs:<http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos:<http://www.w3.org/2004/02/skos/core#>
PREFIX base:<http://www.wikidata.org/ontology#>
PREFIX schema:<http://schema.org/>

SELECT ?tmp1 ?tmp2
WHERE {
    { GRAPH <http://wikidata.org/statements> {
e:Q2877957 e:P19s/e:P19v ?tmp1 .
e:Q2877957 e:P106s/e:P106v ?tmp2 .
    } }

}

LIMIT 2000
    """)
# BIND( ?tmp1 = e:Q1486 as ?v0 ) . 
# BIND( ?tmp2 = e:Q36180 as ?v1 ) .
# BIND( ?v0 || ?v1 as ?out ) .
    import pdb; pdb.set_trace()

    relations = ['occupation', 'sex or gender', 'date of birth', 'country', 'head of state', 'child', 'creator', 'instance of', 'subclass of']
    for relation in relations:
        results = query_of_relation(GetWikiRelatedWords.rel_name2id[relation])
        # results['e1']
        print(len(results))
        print(results[0])
        import pdb; pdb.set_trace()

def test_convert_fn_to_query():
    functions = [
        "mother(0)", "father(mother(0))",
        "union(father(0), mother(0))", "union(child(0), child(0))",
        "union(father(1), mother(0))", "union(father(1), child(0))",
        "union(father(mother(0)), mother(0))", "union(father(mother(1)), mother(0))", 
        "union(union(father(1), mother(0)), mother(father(1)))", "union(union(father(mother(1)), mother(0)), father(mother(0)))"
    ]
    for function in functions:
        print(function)
        queries, var2fxn = convert_fntree2queries(parse_to_function_tree(function))
        print(queries)
        print(var2fxn)
        print()
        # TODO check query?

if __name__ == "__main__":
    # test_function_parse()
    test_wikidata_query()
    # test_convert_fn_to_query()
    # test_convert_fn_to_nl()