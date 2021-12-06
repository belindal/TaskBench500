# TaskBench500

Get data
```bash
wget http://web.mit.edu/bzl/www/TaskBench500/TaskBenchData.tar.gz
tar -xzvf TaskBenchData.tar.gz
```

Get resources for creating data
```bash
wget http://web.mit.edu/bzl/www/TaskBench500/resources.tar.gz
tar -xzvf resources.tar.gz
```

### Dataset Creation Procedure
```bash
python scripts/make_data.py \
    --function "map{translate[eng->spa](0)}(S)" \
    --save_dir <directory to save to> \
    --sample_type [seq|word] \
    --num_samples <\# of samples>
```


Test
```bash
python scripts/make_data.py \
    --function "map{translate[eng->spa](0)}(S)" \
    --save_dir temp \
    --sample_type word \
    --num_samples <\# of samples>
```