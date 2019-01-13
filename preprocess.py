import torchtext.data as data
import torchtext.vocab as vocab
import torch
import time
import random
import numpy as np
from tokenizers import WhitespaceTokenizer, CustomTokenizer
from utils import print_duration
from my_torchtext import MyTabularDataset, MyVectors
from functools import partial


def normal_init(tensor, std):
    return torch.randn_like(tensor) * std


class Data:
    def __init__(self, train_csv, test_csv):
        self.test_csv = test_csv
        self.train_csv = train_csv
        self.text = None
        self.qid = None
        self.train = None
        self.test = None
        self.target = None

    def preprocess(self, tokenizer, var_length=False):
        # types of csv columns
        time_start = time.time()
        self.text = data.Field(batch_first=True, tokenize=tokenizer, include_lengths=var_length)
        self.qid = data.Field()
        self.target = data.Field(sequential=False, use_vocab=False, is_target=True)

        # read and tokenize data
        print('Read and tokenize data')
        self.train = MyTabularDataset(path=train_csv, format='csv',
                                 fields={'qid': ('qid', self.qid),
                                         'question_text': ('text', self.text),
                                         'target': ('target', self.target)})
        self.test = MyTabularDataset(path=test_csv, format='csv',
                                fields={'qid': ('qid', self.qid),
                                        'question_text': ('text', self.text)})
        self.text.build_vocab(self.train, self.test, min_freq=1)
        self.qid.build_vocab(self.train, self.test)
        print_duration(time_start, 'Time to read and tokenize data: ')
        return self.text, self.qid

    def embedding_lookup(self, embedding, cache, unk_std):
        print('Embedding lookup')
        time_start = time.time()
        unk_init = partial(normal_init, std=unk_std)
        self.text.vocab.load_vectors(MyVectors(embedding, cache=cache, unk_init=unk_init))
        print_duration(time_start, 'Time for embedding lookup: ')
        return

    def split(self, args):
        k = args.kfold
        sr = args.split_ratio
        is_test = args.is_test
        if k:
            data_iter = self.train.split_kfold(k, is_test=is_test, random_state=random.getstate())
        else:
            data_iter = self.train.split(sr, random_state=random.getstate())
        return data_iter


def iterate(train, val, test, batch_size):
    train_iter = data.BucketIterator(dataset=train,
                                     batch_size=batch_size,
                                     sort_key=lambda x: x.text.__len__(),
                                     shuffle=True,
                                     sort=False,
                                     sort_within_batch=True)

    val_iter = data.BucketIterator(dataset=val,
                                   batch_size=batch_size,
                                   sort_key=lambda x: x.text.__len__(),
                                   train=False,
                                   sort=False,
                                   sort_within_batch=True)

    test_iter = data.BucketIterator(dataset=test,
                                    batch_size=batch_size,
                                    sort_key=lambda x: x.text.__len__(),
                                    sort=False,
                                    train=False,
                                    sort_within_batch=True)
    return train_iter, val_iter, test_iter
