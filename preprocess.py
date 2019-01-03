import torchtext.data as data
import torchtext.vocab as vocab
import time
import random

from tokenizers import WhitespaceTokenizer, CustomTokenizer
from utils import print_duration
from dataloader import MyTabularDataset


def preprocess(train_csv, test_csv, tokenizer, embedding, cache):
    # types of csv columns
    location = './cachedir'
    time_start = time.time()
    text = data.Field(batch_first=True, tokenize=tokenizer, include_lengths=True)
    qid = data.Field()
    target = data.Field(sequential=False, use_vocab=False, is_target=True)

    # read and tokenize data
    print('Reading data.')
    train = MyTabularDataset(path=train_csv, format='csv',
                             fields={'qid': ('qid', qid),
                                     'question_text': ('text', text),
                                     'target': ('target', target)})
    test = MyTabularDataset(path=test_csv, format='csv',
                            fields={'qid': ('qid', qid),
                                    'question_text': ('text', text)})
    text.build_vocab(train, test, min_freq=1)
    qid.build_vocab(train, test)
    print_duration(time_start, 'Time to read and tokenize data: ')

    # embeddings lookup
    print('Embedding lookup...')
    time_start = time.time()
    text.vocab.load_vectors(vocab.Vectors(embedding, cache=cache))
    print_duration(time_start, 'Time for embedding lookup: ')

    return train, test, text, qid


def split(train, args):
    k = args.kfold
    if k:
        data_iter = train.split_kfold(k, is_test=args.test, random_state=random.getstate())
    else:
        data_iter = train.split(args.split_ratio, random_state=random.getstate())
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
