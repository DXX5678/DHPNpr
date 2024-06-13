import logging
import os
import re

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Example(object):
    """A single training/test example.(Discriminator)"""

    def __init__(self,
                 idx,
                 sequence1,
                 sequence2,
                 label
                 ):
        self.idx = idx
        self.sequence1 = sequence1
        self.sequence2 = sequence2
        self.label = label


class InputFeatures(object):
    """A single training/test features for a example.(Discriminator)"""

    def __init__(self,
                 example_id,
                 sequence1_ids,
                 sequence2_ids,
                 sequence1_mask,
                 sequence2_mask,
                 label
                 ):
        self.example_id = example_id
        self.sequence1_ids = sequence1_ids
        self.sequence2_ids = sequence2_ids
        self.sequence1_mask = sequence1_mask
        self.sequence2_mask = sequence2_mask
        self.label = label


def readLines(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            lines.append(line.strip())
        f.close()
    return lines


def prepare_examples_dir(dir, patch_dir, mode="train"):
    if mode == "train":
        ids_f = os.path.join(dir, "trn.ids")
    else:
        ids_f = os.path.join(dir, "valid.ids")
    return prepare_CR3_examples(ids_f, dir, patch_dir)


def prepare_CR3_examples(ids_f, dir, patch_dir):
    ids = readLines(ids_f)
    examples = []
    idx = 0
    for id in tqdm(ids):
        buggy_methods_dir = os.path.join(dir, "buggy_methods")
        fix_lines_dir = os.path.join(dir, "fix_lines")
        patch_lines_dir = os.path.join(patch_dir, "patch_lines")
        if not os.path.exists(os.path.join(patch_lines_dir, id + ".txt")):
            continue
        fix_line = open(os.path.join(fix_lines_dir, id + ".txt"), 'r', encoding='utf8').read().strip()
        patch_line = open(os.path.join(patch_lines_dir, id + ".txt"), 'r', encoding='utf8').read().strip()
        buggy_method = readLines(os.path.join(buggy_methods_dir, id + ".txt"))
        buggy_code = '\n'.join(buggy_method)
        buggy_code = re.sub('\s+', ' ', buggy_code)
        patch_code = re.sub('\s+', ' ', patch_line.strip())
        fix_code = re.sub('\s+', ' ', fix_line.strip())
        examples.append(Example(
            idx=idx,
            sequence1=buggy_code,
            sequence2=patch_code,
            label=0
        ))
        idx += 1
        examples.append(Example(
            idx=idx,
            sequence1=buggy_code,
            sequence2=fix_code,
            label=1
        ))
        idx += 1
        # print(idx,input,output)
    return examples


def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage, no_share = item
    sequence1_mask, sequence2_mask = None, None
    if args.model_type == "codet5":
        sequence1_str = example.sequence1
        sequence1_str = sequence1_str.replace('</s>', '<unk>')
        sequence1_ids = tokenizer.encode(sequence1_str, max_length=args.max_source_length, padding='max_length',
                                         truncation=True)
        assert sequence1_ids.count(tokenizer.eos_token_id) == 1

        sequence2_str = example.sequence2
        sequence2_str = sequence2_str.replace('</s>', '<unk>')
        sequence2_ids = tokenizer.encode(sequence2_str, max_length=args.max_target_length, padding='max_length',
                                         truncation=True)
        assert sequence2_ids.count(tokenizer.eos_token_id) == 1
    else:
        sequence1_tokens = tokenizer.tokenize(example.sequence1)[:args.max_source_length - 2]
        sequence1_tokens = [tokenizer.cls_token] + sequence1_tokens + [tokenizer.sep_token]
        sequence1_ids = tokenizer.convert_tokens_to_ids(sequence1_tokens)
        sequence1_mask = [1] * (len(sequence1_tokens))
        padding_length = args.max_source_length - len(sequence1_ids)
        sequence1_ids += [tokenizer.pad_token_id] * padding_length
        sequence1_mask += [0] * padding_length

        sequence2_tokens = tokenizer.tokenize(example.sequence2)[:args.max_target_length - 2]
        sequence2_tokens = [tokenizer.cls_token] + sequence2_tokens + [tokenizer.sep_token]
        sequence2_ids = tokenizer.convert_tokens_to_ids(sequence2_tokens)
        sequence2_mask = [1] * len(sequence2_ids)
        padding_length = args.max_target_length - len(sequence2_ids)
        sequence2_ids += [tokenizer.pad_token_id] * padding_length
        sequence2_mask += [0] * padding_length

    return InputFeatures(
        example_id=example.idx,
        sequence1_ids=sequence1_ids,
        sequence2_ids=sequence2_ids,
        sequence1_mask=sequence1_mask,
        sequence2_mask=sequence2_mask,
        label=example.label
    )


def load_and_cache_gen_data(args, dir, patch_dir, pool, tokenizer, split_tag, mode="train", no_share=False,
                            only_src=False, is_sample=False):
    examples = prepare_examples_dir(dir, patch_dir, mode)
    tuple_examples = [(example, idx, tokenizer, args, split_tag, no_share) for idx, example in enumerate(examples)]
    features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
    all_sequence1_ids = torch.tensor([f.sequence1_ids for f in features], dtype=torch.long)
    all_sequence2_ids = torch.tensor([f.sequence2_ids for f in features], dtype=torch.long)
    all_label = torch.tensor([f.label for f in features], dtype=torch.long)
    if no_share:
        data = TensorDataset(all_sequence1_ids, all_sequence2_ids, all_label)
        return examples, data
    if args.model_type == "codet5":
        # cache the data into args.cache_path except it is sampled
        # only_src: control whether to return only source ids for bleu evaluating (dev/test)
        # return: examples (Example object), data (TensorDataset)
        data_tag = '_all'
        cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + ('_src' if only_src else '') + data_tag)
        cache_flag = True

        if os.path.exists(cache_fn) and not is_sample:
            logger.info("Load cache data from %s", cache_fn)
            data = torch.load(cache_fn)
            return examples, data

        all_sequence1_mask = all_sequence1_ids.ne(tokenizer.pad_token_id)
        all_sequence2_mask = all_sequence2_ids.ne(tokenizer.pad_token_id)
        data = TensorDataset(all_sequence1_ids, all_sequence1_mask, all_sequence2_ids, all_sequence2_mask, all_label)

        logger.info("Create cache data into %s", cache_fn)
        if args.local_rank in [-1, 0] and not is_sample and cache_flag:
            torch.save(data, cache_fn)
    else:
        all_sequence1_mask = torch.tensor([f.sequence1_mask for f in features], dtype=torch.long)
        all_sequence2_mask = torch.tensor([f.sequence2_mask for f in features], dtype=torch.long)
        data = TensorDataset(all_sequence1_ids, all_sequence1_mask, all_sequence2_ids, all_sequence2_mask, all_label)
    return examples, data
