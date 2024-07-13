import argparse
import logging
import math
import os
import time

import torch
import torch.nn as nn
import multiprocessing
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import RandomSampler, DistributedSampler, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)
from transformers import get_linear_schedule_with_warmup

from data_preprocess import load_and_cache_gen_data
from utils import get_elapse_time
from configs import set_seed
from model import DiscriminatorShare
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from UniXcoder.model import Seq2Seq

MODEL_CLASSES = {'unixcoder': (RobertaConfig, RobertaModel, RobertaTokenizer)}
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_acc_epoch(args, eval_data, eval_examples, unixcoder, model, loss_fn):
    predictions, true_labels = [], []
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)
    # Start evaluating model
    logger.info("  " + "***** Running acc evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    unixcoder.eval()
    model.eval()
    eval_loss, batch_num = 0, 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval acc"):
        batch = tuple(t.to(args.device) for t in batch)
        sequence_1_ids, sequence_1_mask, sequence_2_ids, sequence_2_mask, labels = batch

        with torch.no_grad():
            sequence_1_output = unixcoder.encoder(sequence_1_ids, attention_mask=sequence_1_mask)
            sequence_2_output = unixcoder.encoder(sequence_2_ids, attention_mask=sequence_2_mask)
            outputs = model(sequence_1_output[0], sequence_2_output[0], sequence_1_mask, sequence_2_mask)
            loss = loss_fn(outputs.view(-1, 2), labels.view(-1))
        # eval_loss += loss.item()
        eval_loss += torch.mean(loss).item()
        batch_num += 1
        softmax = nn.Softmax(dim=-1)
        _, max_index = torch.max(softmax(outputs).view(-1, 2), dim=-1)
        predictions.extend(max_index.view(-1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(true_labels, predictions)
    eval_loss = round(eval_loss / batch_num, 5)
    return eval_loss, accuracy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", default="CodeBERT-finetune")
    ## Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--log_file_dir", default=None, type=str, required=True,
                        help="The output directory where the log_file will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    ## Other parameters
    parser.add_argument("--train_dir", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_dir", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--patch_train_dir", default=None, type=str,
                        help="The generated patches filename for training. Should contain the .jsonl files for this "
                             "task.")
    parser.add_argument("--patch_valid_dir", default=None, type=str,
                        help="The generated patches filename for evaluation. Should contain the .jsonl files for this "
                             "task.")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--cache_path", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to inference.")

    parser.add_argument("--do_defects4j", action='store_true',
                        help="Whether to inference on the Defect4J.")
    parser.add_argument("--buggy_file", default="", type=str,
                        help="Path of the buggy project on the Defect4J.")
    parser.add_argument("--buggy_line", default="", type=str,
                        help="Location of the buggy code.")

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--save_last_checkpoints", default='save', action='store_true')
    parser.add_argument("--always_save_model", default='save', action='store_true')
    # print arguments
    args = parser.parse_args()
    t0 = time.time()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    # import！！！you must set is_decoder as True for generation
    config.is_decoder = True
    encoder = RobertaModel.from_pretrained(args.model_name_or_path, config=config)

    unixcoder = Seq2Seq(encoder=encoder, decoder=encoder, config=config,
                    beam_size=args.beam_size, max_length=args.max_target_length,
                    sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        unixcoder.load_state_dict(torch.load(args.load_model_path))

    model = DiscriminatorShare(config.hidden_size)

    model.to(device)
    unixcoder.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        # unixcoder = torch.nn.DataParallel(unixcoder)
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    log_file = open(os.path.join(args.log_file_dir, 'Discriminator_Share_' + args.model_type + '.log'), 'a+')

    if args.do_train:
        # Prepare training data loader
        train_examples, train_data = load_and_cache_gen_data(args, args.train_dir, args.patch_train_dir, pool,
                                                             tokenizer, 'train', mode="train")
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        # train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)

        # Prepare optimizer and schedule (linear warmup and decay)
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        loss_fn = nn.CrossEntropyLoss()
        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)
        log_file.write("  Start Training...\n")
        dev_dataset = {}
        global_step, best_bleu_em, best_acc, best_loss = 0, -1, 0, 1000
        not_acc_inc_cnt, not_loss_dec_cnt = 0, 0
        min_delta = 1e-4

        for cur_epoch in range(int(args.num_train_epochs)):
            logger.info("  Now epoch = %d", cur_epoch)
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            unixcoder.eval()
            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)
                sequence_1_ids, sequence_1_mask, sequence_2_ids, sequence_2_mask, labels = batch
                with torch.no_grad():
                    sequence_1_output = unixcoder.encoder(sequence_1_ids, attention_mask=sequence_1_mask)
                    sequence_2_output = unixcoder.encoder(sequence_2_ids, attention_mask=sequence_2_mask)
                outputs = model(sequence_1_output[0], sequence_2_output[0], sequence_1_mask, sequence_2_mask)
                loss = loss_fn(outputs.view(-1, 2), labels.view(-1))

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += sequence_1_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))
                    log_file.write("[{}] Train loss {}\n".format(cur_epoch, round(train_loss, 3)))

            if args.do_eval and (int(cur_epoch) % 1 == 0):
                # Eval model with dev dataset
                logger.info("  Start Eval...")
                log_file.write("  Start Eval...\n")
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_dir, args.patch_valid_dir, pool,
                                                                       tokenizer, 'dev', mode='dev')
                    dev_dataset['dev_loss'] = eval_examples, eval_data

                eval_loss, eval_acc = eval_acc_epoch(args, eval_data, eval_examples, unixcoder, model, loss_fn)
                result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_loss': eval_loss, 'eval_acc': eval_acc}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    log_file.write("  %s = %s\n" % (key, str(result[key])))
                logger.info("  " + "*" * 20)
                log_file.write("  End Eval...\n")
                # if args.data_num == -1:
                #     tb_writer.add_scalar('dev_ppl', eval_ppl, cur_epoch)
                if eval_loss < best_loss:
                    not_loss_dec_cnt = 0
                    best_loss = eval_loss
                else:
                    not_loss_dec_cnt += 1
                    logger.info("loss does not decrease for %d epochs", not_loss_dec_cnt)
                # save last checkpoint
                if args.save_last_checkpoints:
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the last model into %s", output_model_file)

                if eval_acc > best_acc + min_delta:
                    not_acc_inc_cnt = 0
                    logger.info("  Best acc:%s", eval_acc)
                    logger.info("  " + "*" * 20)
                    log_file.write("[%d] Best acc changed into %.4f\n" % (cur_epoch, eval_acc))
                    best_acc = eval_acc

                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if args.always_save_model:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the best acc model into %s", output_model_file)
                else:
                    not_acc_inc_cnt += 1
                    logger.info("acc does not increase for %d epochs", not_acc_inc_cnt)
                if any([x > args.patience for x in [not_loss_dec_cnt, not_acc_inc_cnt]]):
                    early_stop_str = "[%d] Early stop as not_acc_inc_cnt=%d and as not_loss_dec_cnt=%d\n" % (
                        cur_epoch, not_acc_inc_cnt, not_loss_dec_cnt)
                    logger.info(early_stop_str)
                    log_file.write(early_stop_str)
                    break
        logger.info("Finish training and take %s", get_elapse_time(t0))

    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    log_file.write("Finish and take {}\n".format(get_elapse_time(t0)))
    log_file.close()


if __name__ == "__main__":
    main()
