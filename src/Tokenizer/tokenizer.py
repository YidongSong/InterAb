# -*- coding: utf-8 -*-

import tempfile
import pandas as pd
import numpy as np
from abc import abstractmethod
from os.path import join
from transformers import BertTokenizer , AutoTokenizer


class CommonTokenizer(object):
    def __init__(self, logger, tokenizer_name='common', add_hyphen=False):
        self.PAD = "$"
        self.MASK = "."
        self.UNK = "?"
        self.SEP = "|"
        self.CLS = "*"

        self.logger = logger
        self.logger.info(f'Using {tokenizer_name} tokenizer.')

        self.token_with_special_list, self.token2index_dict = self._get_vocab_dict(add_hyphen)
    
    def _get_vocab_dict(self, add_hyphen=False):
        amino_acids_list = [c for c in 'ACDEFGHIKLMNPQRSTVWY']
        special_tokesn = [self.PAD, self.MASK, self.UNK, self.SEP, self.CLS]

        if add_hyphen:
            self.logger.info('Add hyphen - in the tokenizer')
            token_list = ['-'] + amino_acids_list + special_tokesn
        else:
            token_list = amino_acids_list + special_tokesn
        token2index_dict = {t: i for i, t in enumerate(token_list)}

        return token_list, token2index_dict

    def get_bert_tokenizer(self, max_len=64, tokenizer_dir=None):
        if tokenizer_dir is not None:
            self.logger.info('Loading pre-trained tokenizer...')
            tok = BertTokenizer.from_pretrained(
                tokenizer_dir,
                do_lower_case=False,
                do_basic_tokenize=True,
                tokenize_chinese_chars=False,
                pad_token=self.PAD,
                mask_token=self.MASK,
                unk_token=self.UNK,
                sep_token=self.SEP,
                cls_token=self.CLS,
                padding_side="right"
                )
            return tok

        with tempfile.TemporaryDirectory() as tempdir:
            vocab_fname = self._write_vocab(self.token2index_dict, join(tempdir, "vocab.txt"))
            tok = BertTokenizer(
                vocab_fname,
                do_lower_case=False,
                do_basic_tokenize=True,
                tokenize_chinese_chars=False,
                pad_token=self.PAD,
                mask_token=self.MASK,
                unk_token=self.UNK,
                sep_token=self.SEP,
                cls_token=self.CLS,
                model_max_len=max_len,
                padding_side="right"
                )
        return tok

    def split(self, seq):
        return list(seq)

    def _write_vocab(self, vocab, fname):
        """
        Write the vocabulary to the fname, one entry per line
        Mostly for compatibility with transformer AutoTokenizer
        """
        with open(fname, "w") as sink:
            for v in vocab:
                sink.write(v + "\n")
        return fname 


def get_tokenizer(tokenizer_name, add_hyphen, logger, vocab_dir, token_length_list=[2,3]):
    if tokenizer_name == 'common':
        MyTokenizer = CommonTokenizer(logger=logger, add_hyphen=add_hyphen)

    return MyTokenizer