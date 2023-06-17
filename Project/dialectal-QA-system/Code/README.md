
### Requirements and installation 
```
./install.sh
```

#### Training baseline:

``` 

source vdial/bin/activate


python src/run_squad.py \
	--model_type bert \
	--model_name_or_path=bert-base-multilingual-uncased \
	--do_train \
	--do_lower_case \
	--train_file 'data/dialqa-train.json' \
	--per_gpu_train_batch_size 16 \
	--per_gpu_eval_batch_size 24 \
	--learning_rate 3e-5 \
	--num_train_epochs 3 \
	--max_seq_length 384 \
	--doc_stride 128 \
	--output_dir 'train_cache_output/' \
	--overwrite_cache \
	--overwrite_output_dir
```

#### Prediction on augmented dev data

```
python src/run_squad.py \
	--model_type bert \
	--model_name_or_path='train_cache_output' \
	--do_eval \
	--do_lower_case \
	--predict_file 'data/dialqa-dev-aug.json' \
	--per_gpu_train_batch_size 16 \
	--per_gpu_eval_batch_size 16 \
	--learning_rate 3e-5 \
	--num_train_epochs 3 \
	--max_seq_length 384 \
	--doc_stride 128 \
	--output_dir 'outputs/aug-mbert' \
	--overwrite_output_dir
``` 


#### Prediction on test data

```
python src/run_squad.py \
	--model_type bert \
	--model_name_or_path='train_cache_output' \
	--do_eval \
	--do_lower_case \
	--predict_file 'data/dialqa-test.json' \
	--per_gpu_train_batch_size 16 \
	--per_gpu_eval_batch_size 16 \
	--learning_rate 3e-5 \
	--num_train_epochs 3 \
	--max_seq_length 384 \
	--doc_stride 128 \
	--output_dir 'outputs/test-mbert' \
	--overwrite_output_dir
``` 
