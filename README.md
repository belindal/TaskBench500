# TaskBench500
The TaskBench500 dataset and code for generating tasks.

## Data
The TaskBench dataset is available under
```bash
wget http://web.mit.edu/bzl/www/TaskBench500/TaskBenchData.tar.gz
tar -xzvf TaskBenchData.tar.gz
```

The file structure looks as follows:
    .
    ├── atomic                      # atomic tasks
    ├── seq_composite               # sequential compositional tasks
    │   ├── filter                  # tasks of the form filter{λx.F(x)}(S)
    │   ├── map                     # tasks of the form map{λx.F(x)}(S)
    │   └── mapfilter               # tasks of the form map{λx.F(x)}(filter{λx.F(x)}(S))
    └── word_composite              # word-level compositional tasks
        ├── chaining                # tasks of the form F'(F)
        ├── intersection            # tasks of the form F∩F'
        ├── land                    # tasks of the form F∧F'
        ├── lor                     # tasks of the form F∨F'
        └── union                   # tasks of the form F∪F'

where F and F' are arbitrary functions.

## Dataset Creation Procedure
Coming soon.

## Training a Model
Coming soon.