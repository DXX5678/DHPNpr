import os
import re
import random
import logging
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 buggy_method,
                 source,
                 target,
                 label,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.buggy_method = buggy_method
        self.source = source
        self.target = target
        self.label = label
        self.url = url
        self.task = task
        self.sub_task = sub_task


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 buggy_method_ids,
                 source_ids,
                 target_ids,
                 label,
                 url=None
                 ):
        self.example_id = example_id
        self.buggy_method_ids = buggy_method_ids
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.label = label
        self.url = url


def readLines(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            lines.append(line.strip())
        f.close()
    return lines


def prepare_examples_dir(dir, mode="train"):
    if mode == "train":
        ids_f = os.path.join(dir, "trn.ids")
    else:
        ids_f = os.path.join(dir, "valid.ids")
    buggy_methods_dir = os.path.join(dir, "buggy_methods")
    buggy_lines_dir = os.path.join(dir, "buggy_lines")
    fix_lines_dir = os.path.join(dir, "fix_lines")
    return prepare_CR3_examples(ids_f, buggy_methods_dir, buggy_lines_dir, fix_lines_dir)


def prepare_CR3_examples(ids_f, buggy_methods_dir, buggy_lines_dir, fix_lines_dir):
    ids = readLines(ids_f)
    examples = []
    idx = 0
    for id in tqdm(ids):
        buggy_code = readLines(os.path.join(buggy_methods_dir, id + ".txt"))
        buggy_code = '\n'.join(buggy_code)
        buggy_code = re.sub('\s+', ' ', buggy_code)
        buggy_line = open(os.path.join(buggy_lines_dir, id + ".txt"), 'r', encoding='utf8').read().strip()
        fix_line = open(os.path.join(fix_lines_dir, id + ".txt"), 'r', encoding='utf8').read().strip()
        buggy_method = readLines(os.path.join(buggy_methods_dir, id + ".txt"))
        for ind in range(len(buggy_method)):
            if buggy_line in buggy_method[ind]:
                buggy_method[ind] = " <BUGS> " + buggy_line + " <BUGE> "
        input = '\n'.join(buggy_method)
        input = re.sub('\s+', ' ', input)
        output = re.sub('\s+', ' ', " <FIXS> " + fix_line.strip() + " <FIXE> ")
        examples.append(Example(
            idx=idx,
            buggy_method=buggy_code,
            source=input,
            target=output,
            label=1
        ))
        idx += 1
        # print(idx,input,output)

    return examples


def calc_stats(examples, tokenizer=None, is_tokenize=False):
    avg_src_len = []
    avg_trg_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    for ex in examples:
        if is_tokenize:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(ex.source)))
            avg_trg_len_tokenize.append(len(tokenizer.tokenize(str(ex.target))))
        else:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
    if is_tokenize:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))
        logger.info("[TOKENIZE] avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    np.mean(avg_src_len_tokenize), np.mean(avg_trg_len_tokenize), max(avg_src_len_tokenize),
                    max(avg_trg_len_tokenize))
    else:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))


def add_lang_by_task(target_str, task, sub_task):
    if task == 'summarize':
        target_str = '<en> ' + target_str
    elif task == 'refine':
        target_str = '<java> ' + target_str
    elif task == 'translate':
        if sub_task == 'java-cs':
            target_str = '<c_sharp> ' + target_str
        else:
            target_str = '<java> ' + target_str
    elif task == 'concode':
        target_str = '<java> ' + target_str
    elif task == 'defect':
        target_str = target_str
    return target_str


def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item
    buggy_method_str = example.buggy_method
    buggy_method_str = buggy_method_str.replace('</s>', '<unk>')
    buggy_method_ids = tokenizer.encode(buggy_method_str, max_length=args.max_source_length, padding='max_length',
                                        truncation=True)
    assert buggy_method_ids.count(tokenizer.eos_token_id) == 1
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        if args.sub_task != 'none':
            source_str = "{} {}: {}".format(args.task, args.sub_task, example.source)
        else:
            source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source

    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    assert source_ids.count(tokenizer.eos_token_id) == 1
    if stage == 'test':
        target_ids = []
    else:
        target_str = example.target
        if args.add_lang_ids:
            target_str = add_lang_by_task(example.target, args.task, args.sub_task)
        if args.task in ['defect', 'clone']:
            if target_str == 0:
                target_str = 'false'
            elif target_str == 1:
                target_str = 'true'
            else:
                raise NameError
        target_str = target_str.replace('</s>', '<unk>')
        target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',
                                      truncation=True)
        assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        example_index,
        buggy_method_ids,
        source_ids,
        target_ids,
        example.label,
        url=example.url
    )


def load_and_cache_gen_data(args, dir, pool, tokenizer, split_tag, mode="train", only_src=False, is_sample=False):
    # cache the data into args.cache_path except it is sampled
    # only_src: control whether to return only source ids for bleu evaluating (dev/test)
    # return: examples (Example object), data (TensorDataset)
    data_tag = '_all' if args.data_num == -1 else '_%d' % args.data_num
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + ('_src' if only_src else '') + data_tag)
    cache_flag = True
    examples = prepare_examples_dir(dir, mode)

    if is_sample:
        examples = random.sample(examples, min(5000, len(examples)))
    if split_tag == 'train':
        calc_stats(examples, tokenizer, is_tokenize=True)
    else:
        calc_stats(examples)
    if os.path.exists(cache_fn) and not is_sample:
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 5k data for computing bleu from %s", dir)
        else:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
        features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_buggy_method_ids = torch.tensor([f.buggy_method_ids for f in features], dtype=torch.long)
        all_buggy_method_mask = all_buggy_method_ids.ne(tokenizer.pad_token_id)
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        if split_tag == 'test' or only_src:
            data = TensorDataset(all_source_ids)
            cache_flag = False
        else:
            all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
            all_source_mask = all_source_ids.ne(tokenizer.pad_token_id)
            all_target_mask = all_target_ids.ne(tokenizer.pad_token_id)
            data = TensorDataset(all_buggy_method_ids, all_buggy_method_mask, all_source_ids, all_source_mask,
                                 all_target_ids, all_target_mask, all_label)
        if args.local_rank in [-1, 0] and not is_sample and cache_flag:
            torch.save(data, cache_fn)
    return examples, data
