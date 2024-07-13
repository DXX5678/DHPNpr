
lr=1e-4
batch_size=16 #32
beam_size=5
max_source_length=512
max_target_length=256
epoch=30

load_model_path=/home/dxx/DHPNpr/saved_models/UniXcoder_256/checkpoint-best-ppl/pytorch_model.bin
output_dir=/home/dxx/DHPNpr/saved_models/Discriminator_share_unixcoder
log_file=train.log
train_file=/home/dxx/DHPNpr/data/Train
validate_file=/home/dxx/DHPNpr/data/Valid
patch_train_dir=/home/dxx/DHPNpr/data_dis/UniXcoder/Train
patch_valid_dir=/home/dxx/DHPNpr/data_dis/UniXcoder/Valid
model_name_or_path=/home/dxx/DHPNpr/UniXcoder/unixcoder-base
tokenizer_name=/home/dxx/DHPNpr/UniXcoder/unixcoder-base
cache_path=$output_dir/cache_data
log_file_dir=/home/dxx/DHPNpr/logging

mkdir -p $output_dir

python ./Discriminator/run_share_UniXcoder.py \
--do_train \
--do_eval \
--model_type unixcoder \
--model_name_or_path $model_name_or_path \
--tokenizer_name $tokenizer_name \
--load_model_path $load_model_path \
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
--task_name Discriminator_share_unixcoder \
2>&1| tee $output_dir/$log_file