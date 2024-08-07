
lr=1e-4
batch_size=16 #32
beam_size=5
max_source_length=512
max_target_length=256
epoch=30


output_dir=/home/dxx/DHPNpr/saved_models/Discriminator_no_shareS_codet5
log_file=train.log
train_file=/home/dxx/DHPNpr/data/Train
validate_file=/home/dxx/DHPNpr/data/Valid
patch_train_dir=/home/dxx/DHPNpr/data_dis/CodeT5/Train
patch_valid_dir=/home/dxx/DHPNpr/data_dis/CodeT5/Valid
model_name_or_path=/home/dxx/DHPNpr/CodeT5/codet5-base
tokenizer_name=/home/dxx/DHPNpr/CodeT5/codet5-base
cache_path=$output_dir/cache_data
log_file_dir=/home/dxx/DHPNpr/logging

mkdir -p $output_dir

python ./run_no_shareS.py \
--do_train \
--do_eval \
--model_type codet5 \
--model_name_or_path $model_name_or_path \
--tokenizer_name $tokenizer_name \
--train_dir $train_file \
--dev_dir $validate_file \
--patch_train_dir $patch_train_dir \
--patch_valid_dir $patch_valid_dir \
--output_dir $output_dir \
--max_source_length $max_source_length \
--max_target_length $max_target_length \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--num_train_epochs $epoch \
--cache_path $cache_path \
--log_file_dir $log_file_dir \
--task_name Discriminator_no_shareS_codet5 \
2>&1| tee $output_dir/$log_file