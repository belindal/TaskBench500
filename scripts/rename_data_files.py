import json
import os
from glob import glob
import shutil
# from tqdm import tqdm

wiki_entity_save_fn = "resources/wikidata_entity_ids.tsv"
def get_wikidata_entities():
    assert os.path.exists(wiki_entity_save_fn)
    entity_name_to_id = {}
    with open(wiki_entity_save_fn) as f:
        for line in f:
            line = line.strip().split('\t')
            entity_name_to_id[line[1].lower()] = line[0]
    return entity_name_to_id

def check_if_valid(dirname):
    return os.path.exists(dirname.replace("TaskBenchData_orig", "TaskBenchData"))

def check_and_move(dirname, new_dirname):
    if dirname != new_dirname:
        if os.path.exists(new_dirname):
            print(f"Already exists!: {dirname} -> {new_dirname}")
            breakpoint()
        elif not check_if_valid(new_dirname):
            print(f"Name invalid: {dirname} -> {new_dirname}")
            breakpoint()
        shutil.move(dirname, new_dirname)
    elif not check_if_valid(dirname):
        breakpoint()
        print(f"Name invalid: {dirname}")
    
ent_name2ids = get_wikidata_entities()
ent_ids2name = {ent_name2ids[name]: name for name in ent_name2ids}


def parse_atomic(dirname):
    new_dirname = dirname
    if "[eng>eng]" in new_dirname:
        new_dirname = new_dirname.replace("[eng>eng]", "[eng]")
    if "[spa>spa]" in new_dirname:
        new_dirname = new_dirname.replace("[spa>spa]", "[spa]")
    if "[eng>spa]" in new_dirname:
        new_dirname = new_dirname.replace("[eng>spa]", "[eng->spa]")
    if "[spa>eng]" in new_dirname:
        new_dirname = new_dirname.replace("[spa>eng]", "[spa->eng]")
    if "random" in new_dirname:
        randseed = int(new_dirname[new_dirname.find("{")+1:new_dirname.find("}")])
        lang = new_dirname[new_dirname.find("[")+1:max(new_dirname.find(">"),new_dirname.find("]"))]
        new_dirname = os.path.join(os.path.split(new_dirname)[0], f"random{{{randseed}}}[{lang}](0)")
    
    if new_dirname.startswith("in(") or new_dirname.startswith("in["):
        lang = new_dirname[new_dirname.find("[")+1:new_dirname.find("]")]
        value = new_dirname[new_dirname.find("CONST{")+6:]
        value = value[:value.find("}")]
        if "has_sentiment" in new_dirname:
            attr = "sentiment"
            if value == "neutral": value = "none"
            elif value == "positive": value = "pos"
            if value == "negative": value = "neg"
        elif "has_pos" in new_dirname:
            attr = "POS"
            if value == "a": value = "adj"
            elif value == "n": value = "noun"
            elif value == "r": value = "adv"
            elif value == "v": value = "verb"
        new_dirname = f"is{{{attr}={value}}}[{lang}](0)"

    if new_dirname.startswith("wiki{is["):
        attr = new_dirname.split("wiki{is[")[-1]
        attr = attr.split(']')[0]
        value = new_dirname[new_dirname.find("CONST{")+6:]
        value = value[:value.find("}")]
        try:
            value = ent_ids2name[value]
        except:
            breakpoint()
        new_dirname = f"wiki{{is{{{attr}={value}}}(0)}}"
        
    return new_dirname

def parse_word_composite(fn_name):#, composition_type=None):
    if "intersection" in fn_name:
        atomic_1 = fn_name.split("),")[0].split("intersection(")[1].strip()+")"
        atomic_2 = fn_name.split('),')[1].strip()
        new_atomic_1 = parse_atomic(atomic_1)
        new_atomic_2 = parse_atomic(atomic_2)
        new_fn_name = f"intersection({new_atomic_1},{new_atomic_2}"
    elif "union" in fn_name:
        atomic_1 = fn_name.split("),")[0].split("union(")[1].strip()+")"
        atomic_2 = fn_name.split('),')[1].strip()
        new_atomic_1 = parse_atomic(atomic_1)
        new_atomic_2 = parse_atomic(atomic_2)
        new_fn_name = f"union({new_atomic_1},{new_atomic_2}"
        if fn_name.startswith("wiki{"):
            new_fn_name = "wiki{"+new_fn_name
    elif "land" in fn_name:
        atomic_1 = fn_name.split("),")[0].split("land(")[1].strip()+")"
        atomic_2 = fn_name.split('),')[1].strip()
        # breakpoint()
        new_atomic_1 = parse_atomic("wiki{"+atomic_1).replace("wiki{", "")[:-1]
        new_atomic_2 = parse_atomic("wiki{"+atomic_2).replace("wiki{", "")[:-1]
        new_fn_name = f"wiki{{land({new_atomic_1},{new_atomic_2})}}"
    elif "lor" in fn_name:
        atomic_1 = fn_name.split("),")[0].split("lor(")[1].strip()+")"
        atomic_2 = fn_name.split('),')[1].strip()
        new_atomic_1 = parse_atomic("wiki{"+atomic_1).replace("wiki{", "")[:-1]
        new_atomic_2 = parse_atomic("wiki{"+atomic_2).replace("wiki{", "")[:-1]
        new_fn_name = f"wiki{{lor({new_atomic_1},{new_atomic_2})}}"
    elif fn_name.count("(") > 1:
        outer_atomic = fn_name.split("(")[0]
        inner_atomic = fn_name.split("(")[1]
        extra = "(".join(fn_name.split("(")[2:])
        new_outer_atomic = parse_atomic(outer_atomic)
        new_inner_atomic = parse_atomic(inner_atomic)
        new_fn_name = f"{new_outer_atomic}({new_inner_atomic}({extra}"

    return new_fn_name

def parse_seq_composite(fn_name):
    if fn_name.startswith("filter"):
        filter_fn = fn_name.split("filter{")[-1].split("}(S)")[0]
        new_filter_fn = parse_atomic(filter_fn)
        new_fn_name = f"filter{{{new_filter_fn}}}(S)"
    elif fn_name.startswith("map"):
        if "filter" in fn_name:
            filter_fn = fn_name.split("filter{")[-1].split("}(S)")[0]
            new_filter_fn = parse_atomic(filter_fn)
            map_fn = fn_name.split("map{")[-1].split("}(filter")[0]
            new_map_fn = parse_atomic(map_fn)
            new_fn_name = f"map{{{new_map_fn}}}(filter{{{new_filter_fn}}}(S))"
        else:
            map_fn = fn_name.split("map{")[-1].split("}(S)")[0]
            if map_fn.count("(") == 1:
                new_map_fn = parse_atomic(map_fn)
            else:
                new_map_fn = parse_word_composite(map_fn)
            new_fn_name = f"map{{{new_map_fn}}}(S)"
    return new_fn_name


def replace_lines(dirname):
    """
    Replace function names inside the lines
    """
    for fn in os.listdir(dirname):
        fn = os.path.join(dirname, fn)
        gt_fn = fn.replace("TaskBenchData_orig", "TaskBenchData")
        gt_fn = open(gt_fn).readlines()
        if len(gt_fn) == 0: continue
        gt_line = json.loads(gt_fn[0])
        gt_R = gt_line["R"]
        gt_nlR = gt_line["nl_R"]
        inner_fns = gt_line.get("inner_fns", {}).keys()
        new_lines = []
        old_lines = open(fn).readlines()
        for line in old_lines:
            line = json.loads(line)
            line["R"] = gt_R
            line["nl_R"] = gt_nlR
            if "inner_fns" in line:
                if line["inner_fns"].keys() != inner_fns:
                    # replace keyset
                    new_inner_fns = {}
                    for inner_fn in line["inner_fns"].keys():
                        new_inner_fn = parse_atomic(inner_fn)
                        new_inner_fns[new_inner_fn] = line["inner_fns"][inner_fn]
                    breakpoint()
                    line["inner_fns"] = new_inner_fns
                if line["inner_fns"].keys() != inner_fns:
                    breakpoint()
            new_lines.append(json.dumps(line)+"\n")

        if new_lines == old_lines: continue
        with open(fn, "w") as wf:
            for line in new_lines:
                wf.write(line)


for dirname in glob("TaskBenchData_orig/atomic/*"):
    new_dirname = dirname
    if not check_if_valid(dirname):
        parent_dir, fn_name = os.path.split(dirname)
        new_fn_name = parse_atomic(fn_name)
        new_dirname = os.path.join(parent_dir, new_fn_name)
        check_and_move(dirname, new_dirname)

    print(new_dirname)
    replace_lines(new_dirname)

for word_composite_type in glob("TaskBenchData_orig/word_composite/*"):
    for dirname in glob(word_composite_type+"/*"):
        new_dirname = dirname
        if not check_if_valid(dirname):
            parent_dir, fn_name = os.path.split(dirname)
            new_fn_name = parse_word_composite(fn_name)#, os.path.split(word_composite_type)[-1])
            new_dirname = os.path.join(parent_dir, new_fn_name)
            check_and_move(dirname, new_dirname)

        print(new_dirname)
        replace_lines(new_dirname)

for seq_composite_type in glob("TaskBenchData_orig/seq_composite/*"):
    for dirname in glob(seq_composite_type+"/*"):
        new_dirname = dirname
        if not check_if_valid(dirname):
            parent_dir, fn_name = os.path.split(dirname)
            print(fn_name)
            new_fn_name = parse_seq_composite(fn_name)#, os.path.split(word_composite_type)[-1])
            new_dirname = os.path.join(parent_dir, new_fn_name)
            check_and_move(dirname, new_dirname)

        print(new_dirname)
        replace_lines(new_dirname)

# for dirname in glob("TaskBenchData_orig/seq_composite/*"):
#     parent_dir, fn_name = os.path.split(dirname)
#     new_fn_name = parse_atomic(fn_name)
#     new_dirname = os.path.join(parent_dir, new_fn_name)
#     check_and_move(dirname, new_dirname)



# for dir in glob("TaskBenchData_orig/atomic/*"):
#     shutil.move("")