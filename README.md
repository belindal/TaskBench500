# TaskBench500


### Dataset Creation Procedure
```bash
python scripts/make_data.py \
    --function "map{translate[eng>spa](0)}(S)" \
    --save_dir <directory to save to> \
    --sample_type [seq|word] \
    --num_samples <\# of samples>
```


Test
```bash
python scripts/make_data.py \
    --function "map{translate[eng>spa](0)}(S)" \
    --save_dir temp \
    --sample_type word \
    --num_samples <\# of samples>
```


```bash
python scripts/make_data.py \
    --function "0[eng]" \
    --save_dir temp \
    --sample_type word \
    --num_samples <\# of samples>
```