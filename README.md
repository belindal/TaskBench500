# TaskBench500
The TaskBench500 dataset and code for generating custom tasks and adapting a model to them.

## Data
The TaskBench dataset is available under
```bash
wget dl.fbaipublicfiles.com/task_bench/taskbench500.tar.gz
tar -xzvf taskbench500.tar.gz
```

The file structure looks as follows:

    .
    ├── atomic                      # atomic tasks
    ├── seq_composite               # sequential compositional tasks
    │   ├── filter                  # tasks of the form filter{λx.F(x)}(S)
    │   ├── map                     # tasks of the form map{λx.F(x)}(S)
    │   └── mapfilter               # tasks of the form map{λx.F(x)}(filter{λx.F(x)}(S))
    └── word_composite              # word-level compositional tasks
        ├── chaining                # tasks of the form F(F')
        ├── intersection            # tasks of the form F∩F'
        ├── land                    # tasks of the form F∧F'
        ├── lor                     # tasks of the form F∨F'
        └── union                   # tasks of the form F∪F'

where `F` and `F'` are arbitrary word-level functions.


## Environment Setup
If you wish the run any part of the code (dataset creation, model adaptation), you should set up your environment as follows:
```bash
conda create -n taskbench PYTHON=3.9
conda activate taskbench
```

Follow instructions here to install pytorch: https://pytorch.org/get-started/locally/

Then install the rest of the packages by running:
```bash
pip install -r requirements.txt
```

To create lexical tasks, you should install multi-lingual wordnet from NLTK. Open an interactive python session and run
```python
import nltk
nltk.download('omw-1.4')
```

Before running any of the commands below, be sure to run
```bash
export PYTHONPATH=.
```


## Generating Datasets for Custom Tasks
Begin by downloading the necessary resources for the creation of these tasks.
```bash
wget dl.fbaipublicfiles.com/task_bench/resources.tar.gz
tar -xzvf resources.tar.gz
```

### Wikidata Setup for Creating Knowledge Tasks
This section describes how to set up Wikidata endpoint to create factual tasks. If you do not need to create tasks which require factual knowledge, you may skip this section.

These instructions were adapted from [the following GIT repository](https://github.com/UKPLab/coling2018-graph-neural-networks-question-answering/blob/master/WikidataHowTo.md).


1. Install and the [opensource version of Virtuoso database server](http://vos.openlinksw.com/owiki/wiki/VOS/VOSDownload#GNU%2FLinux).
```bash
# for Linux
wget https://github.com/openlink/virtuoso-opensource/releases/download/v7.2.7/virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz
tar -xzvf virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz

# for other OS, locate the OS-specific download links in http://vos.openlinksw.com/owiki/wiki/VOS/VOSDownload
```
2. After untar the downloaded binary, the main folder is `virtuoso-opensource`.
3. Download the [pre-built DB file](https://public.ukp.informatik.tu-darmstadt.de/wikidata-dump/wikidata-virtuoso-dump-2017.zip) and extract into `virtuoso-opensource/database`.  The zipped DB file contains two files: `virtuoso.db` and `virtuoso.ini`.
<!-- 4. Change the paths in `virtuoso.ini`, such as `DatabaseFile`, `ErrorLogFile`, etc.  I essentially changed all the paths from relative paths to absolute paths, although I am not sure whether that’s really necessary.  You can also change the HTTPServer port (default:8890) and other settings. -->
4. By default the server runs on port 8890. You may change this by changing `ServerPort` and `DefaultHost` in `virtuoso.ini`. You should also change L11 in `function/wikidata_fns.py` in this repository to point to the Virtuoso server.
5. Start the server by running:
```bash
virtuoso-opensource/bin/virtuoso-t -f -c virtuoso-opensource/database/virtuoso.ini
```
This may take up to a few hours. The starting logs (shown on terminal) look like the following:


### Creating Datasets for Custom Functions
To create a dataset of `(input, output)` pairs corresponding to a particular function specification, run
```bash
python scripts/make_data.py \
    --function <function> \
    --save_dir <output_dataset_directory> \
    --sample_type [word|seq] \
    (--num_samples <number_of_samples>)
```
* `function` takes in a string representation of the task you wish to generate. See below for further guidance on how this is formatted.
* `save_dir` specifies where the output dataset will be saved. The script will save to a file `<save_dir>/<function>.jsonl`.
* `sample_type` specifies how the input will be constructed -- whether the inputs will be single words/entities (`word`) or sequences of words/entities (`seq`).
* `num_samples` refers to how many (sequential) inputs will be sampled for the final dataset. This only applicable if `sample_type` is `seq`, as the entire vocabulary (up to 10k words/entities) is used as input for word-wise tasks.

For bulk processing of functions, you can alternatively pass in a filepath to `--function_file` (instead of specifying a single function to `--function`), where `function_file` is a file consisting of a list of functions separated by newlines. For example, passing a function file containing:
```
antonyms[eng](0)
filter{is{POS=noun}[eng](0)}(S)
wiki{mother(creator(0))}
```
to `make_data.py` script is equivalent to running the script 3 times, once for each function, using `--function`.


### Function Syntax
The syntax for specifying a function looks like
```
function_name{X1,X2,...}[Y1,Y2,...](Z1,Z2,...)
```
<!-- where `args1` i, `args2`   -->
* arguments in braces and brackets (`X1,X2,Y1,Y2`) represent particular features of the function, such as input language, random seed, the per-token operation in the case of `map` and `filter` functions, etc.
* arguments in parentheses (`Z1,Z2`) represent inputs to the function. They can be functions themselves in the case of function composition. By default, `0` is used to represent the input for word-wise functions, and `S` is used to represent the input for sequential functions.

All functions requiring Wikidata are specified within a `wiki{...}` brace, which tells the function creation script that the entire portion within the braces must be converted to a SPARQL query. All function

Some examples of functions include:
```
antonyms[eng](0)
wiki{child(0)}
is{POS=noun}[eng](0)
antonyms[eng](entailments[eng](0))
wiki{mother(creator(0))}
union(antonyms[eng](0),entailments[eng](0))
filter{is{POS=noun}[eng](0)}(S)
map{synonyms[eng](0)}(filter{is{POS=adj}[eng](0)}(S))
```
See Appendix Tables 4-8 in [the paper](https://arxiv.org/pdf/2112.03204.pdf) for further examples of functions.


## Training a Model
The code for adapting models to TaskBench tasks can be found in `run_models/*`.

To fine-tune or prompt-tune a model, run
```bash
python run_models/ft_model.py \
    --data <path_to_data_directory> \
    --batchsize <batchsize> \
    --arch t5-base --lr [0.001|1.0] \
    --epochs <epochs> --patience <patience> \
    --eval_interval <eval_interval> \
    --valid_metric [accuracy|token_accuracy] \
    --seed <random_seed> \
    (--train_size <max_num_training_samples_to_use>)
    (--do_prompt_tune --n_prefix_tokens <num_prompt_tokens_to_use>)
```
The data directory should be comprised of 3 files: `train.jsonl`, `dev.jsonl`, and `test.jsonl`.

We use `accuracy` as the valid_metric for wordwise tasks, and `token_accuracy` for sequential tasks.

By default, the script will perform fine-tuning. To do prompt-tuning, add the `--do_prompt_tune` flag and specify the number of prefix tokens using `--n_prefix_tokens <n>`. We found that the optimal learning rate on our tasks is 0.001 for fine-tuning and 1.0 for prompt-tuning.


<!-- We also provide an training script that can be used to adapt to tasks in bulk. The script can be run using -->
<!-- with hyperparameters pre-specified, which can be run using -->
<!-- ```bash
bash scripts/run_ft_script.py \
    --data_dir <path_to_data_superdirectory> \
    --task_results_file <filepath_containing_tasks_to_evaluate> \
    --num_fewshot <number_of_training_samples_to_use> \
    (--do_prompt_tune --n_prefix_tokens <num_prompt_tokens_to_use>)
```
* `--data_dir` should be comprised of subdirectories for each task, where each task-specific subdirectory contains `train.jsonl`, `dev.jsonl`, and `test.jsonl`.
* `--task_results_file` should contain a list of 
* `--num_fewshot` -->

## Citation
```
@misc{https://doi.org/10.48550/arxiv.2112.03204,
  doi = {10.48550/ARXIV.2112.03204},
  url = {https://arxiv.org/abs/2112.03204},
  author = {Li, Belinda Z. and Yu, Jane and Khabsa, Madian and Zettlemoyer, Luke and Halevy, Alon and Andreas, Jacob},
  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Quantifying Adaptability in Pre-trained Language Models with 500 Tasks},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
