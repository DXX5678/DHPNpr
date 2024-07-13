
lr=5e-5
batch_size=16 #32
beam_size=5
max_source_length=512
max_target_length=512
train_steps=400000
eval_steps=3000


output_dir=/home/dxx/DHPNpr/saved_models/CodeBert_Gan_No_share
log_file=train.log
train_file=/home/dxx/DHPNpr/data/Train
validate_file=/home/dxx/DHPNpr/data/Valid
model_name_or_path=/home/dxx/DHPNpr/CodeBert/codebert-base
tokenizer_name=/home/dxx/DHPNpr/CodeBert/codebert-base
log_file_dir=/home/dxx/DHPNpr/logging
load_model_path=/home/dxx/DHPNpr/saved_models/CodeBert/checkpoint-best-ppl/pytorch_model.bin
load_dis_model_path=/home/dxx/DHPNpr/saved_models/Discriminator_no_share_codebert/checkpoint-best-acc/pytorch_model.bin

mkdir -p $output_dir

python ./GAN/CodeBert/run.py \
--do_train \
--do_eval \
--no_share 1 \
--model_type roberta \
--model_name_or_path $model_name_or_path \
--tokenizer_name $tokenizer_name \
--load_model_path $load_model_path \
--load_dis_model_path $load_dis_model_path \
--train_dir $train_file \
--dev_dir $validate_file \
--output_dir $output_dir \
--max_source_length $max_source_length \
--max_target_length $max_target_length \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--train_steps $train_steps \
--eval_steps $eval_steps \
--log_file_dir $log_file_dir \
2>&1| tee $output_dir/$log_file