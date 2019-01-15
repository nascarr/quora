#!/usr/bin/env python
import numpy as np
import os
import pandas as pd
import torch
import time
import random
import pickle
import os
from functools import partial
import torchtext.data as data
import subprocess
import pandas as pd
import numpy as np
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import glob
import shutil
import torch.optim as optim
import re
import spacy
import torch
import torch.nn as nn
from torchtext.data.dataset import *
from torchtext.vocab import *
import numpy as np
            # Explicitly splitting on " " is important, so we don't
import numpy as np
import torch
import torch.nn as nn
import os
from sklearn.metrics import f1_score
import warnings
import shutil
import time
import subprocess
import sys
import argparse
import os
import random
import sys
import torch.nn as nn
import torch.optim as optim
import argparse
import pandas as pd
import os
import csv
from sklearn.metrics import f1_score
from gensim.models import KeyedVectors
# list of possible models for ensemble


model_dict = {
    'wnews': ('Jan_10_2019__21:56:52', '-es 3 -e 10 -em wnews  -hd 150 -we 10 --lrstep 10'), # 827
    'glove': ('Jan_10_2019__22:10:15', '-es 2 -e 9 -hd 150'), # 829
    'paragram': ('Jan_11_2019__11:10:57', '-hd 150 -es 2 -e 10 -em paragram -t lowerspacy -us 0'), #873
    'gnews': ('Jan_11_2019__20:21:01', '-em gnews -es 2 -e 10 -hd 150 -us 0.1'), # 889
    'wnews_test': ('Jan_10_2019__19:33:05_test', '--mode test -em wnews'),
    'glove_test': ('Jan_10_2019__19:34:39_test', '--mode test -em glove'),
}


def ens_mean(preds, args):
    return np.mean(np.array(preds), 0)


def ens_weight(preds, args):
    weights = args.weights
    preds = np.array(preds)
    if weights:
        weights = np.array(weights)
        preds = np.transpose(preds)
        final_pred = np.transpose(preds.dot(weights))
        return final_pred
    else:
        pass
    return np.mean(np.array(preds), 0)


methods = {'mean': ens_mean,
           'weight': ens_weight}

def read_datasets_to_df(datasets):
    dfs = []
    for d in datasets:
        dfs.append(pd.read_csv(d))
    return dfs


def dfs_to_csv(dfs, csv_names):
    if len(dfs) != len(csv_names):
        raise Exception('len(dfs) != len(csv_names)')

    for df, name in zip(dfs, csv_names):
        df.to_csv(name)
    return csv_names


def small_ds_paths(paths, new_dir, n, string):
        """
        :param paths:
        :param new_dir:
        :param n:
        :param string:
        :return: Paths for small datasets
        """
        small_paths = []

        for p in paths:
            new_path = change_dir_in_path(p, new_dir)
            root, ext = os.path.splitext(new_path)
            small_path = f'{root}_{string}{n}{ext}'
            small_paths.append(small_path)
        return small_paths


def change_dir_in_path(path, new_dir):
    _, file_name = os.path.split(path)
    new_path = os.path.join(new_dir, file_name)
    return new_path



def reduce_datasets(csv_names, new_dir, n, rand=False, seed=None):
    dfs = read_datasets_to_df(csv_names)
    exists = True
    if rand:
        seed = 42 if None else seed
        small_paths = small_ds_paths(csv_names, new_dir, n, 'rand')
        for sp in small_paths:
            if not os.path.exists(sp):
                exists = False
        if not exists:
            small_dfs = [df.sample(n) for df in dfs]
            dfs_to_csv(small_dfs, small_paths)

    else:
        small_paths = small_ds_paths(csv_names, new_dir, n, 'head')
        for sp in small_paths:
            if not os.path.exists(sp):
                exists = False
        if not exists:
            small_dfs = [df.head(n) for df in dfs]
            dfs_to_csv(small_dfs, small_paths)
    return small_paths


def reduce_embedding(emb_path, new_dir, n):
    small_emb_path = small_ds_paths([emb_path], new_dir, n, 'head')[0]
    if not os.path.exists(small_emb_path):
        with open(emb_path, 'r') as f:
            head = [next(f) for i in range(n)]
        with open(small_emb_path, 'w') as f:
            f.write(''.join(head))
    return small_emb_path




def normal_init(tensor, std):
    return torch.randn_like(tensor) * std


class Data:
    def __init__(self, train_csv, test_csv, cache):
        self.test_csv = test_csv
        self.train_csv = train_csv
        self.cache = cache
        self.text = None
        self.qid = None
        self.train = None
        self.test = None
        self.target = None

    def preprocess(self, tokenizer_name, var_length=False):
        # types of csv columns
        time_start = time.time()
        tokenizer = choose_tokenizer(tokenizer_name)
        self.text = data.Field(batch_first=True, tokenize=tokenizer, include_lengths=var_length)
        self.qid = data.Field()
        self.target = data.Field(sequential=False, use_vocab=False, is_target=True)

        # read and tokenize data
        print('read and tokenize data...')
        self.train = MyTabularDataset(path=self.train_csv, format='csv',
                                 fields={'qid': ('qid', self.qid),
                                         'question_text': ('text', self.text),
                                         'target': ('target', self.target)})

        self.test = MyTabularDataset(path=self.test_csv, format='csv',
                                fields={'qid': ('qid', self.qid),
                                        'question_text': ('text', self.text)})
        self.text.build_vocab(self.train, self.test, min_freq=1)
        self.qid.build_vocab(self.train, self.test)
        print_duration(time_start, 'time to read and tokenize data: ')


    def embedding_lookup(self, embedding, unk_std, max_vectors, to_cache):
        print('embedding lookup...')
        time_start = time.time()
        unk_init = partial(normal_init, std=unk_std)
        self.text.vocab.load_vectors(MyVectors(embedding, cache=self.cache, to_cache=to_cache, unk_init=unk_init, max_vectors=max_vectors))
        print_duration(time_start, 'time for embedding lookup: ')
        return

    def split(self, kfold, split_ratio, is_test, seed):
        random.seed(seed)
        if kfold:
            data_iter = self.train.split_kfold(kfold, is_test=is_test, random_state=random.getstate())
        else:
            data_iter = self.train.split(split_ratio, random_state=random.getstate())
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


def _submit(test_ids, predictoins, subm_name):
    sub_df = pd.DataFrame()
    sub_df['qid'] = test_ids
    sub_df['prediction'] = predictoins
    sub_df.to_csv(subm_name, index=False)


def submit(test_ids, labels, probs, subm_name='submission.csv'):
    _submit(test_ids, labels, subm_name)
    _submit(test_ids, probs, 'test_probs.csv')
    print(f'predictions saved in {subm_name}, test_probs.csv file')


def f1_metric(tp, n_targs, n_preds):
    if n_preds == 0 or n_targs == 0 or tp == 0:
        f1 = 0
    else:
        prec = tp/n_preds
        rec = tp/n_targs
        f1 = 2 * rec * prec / (rec + prec)
    return f1


def print_duration(time_start, message):
    time_end = time.time()
    seconds = int(time_end - time_start)
    tr_time = timedelta(seconds=seconds)
    print(f'{message}{tr_time}')
    minutes = seconds/60
    return minutes


def get_hash():
    hash = subprocess.check_output(['git', 'describe', '--always'])
    hash = hash.decode("utf-8")[1:-1]
    return hash


def str_date_time():
    struct_time = time.localtime()
    date_time = time.strftime('%b_%d_%Y__%H_%M_%S', struct_time)
    return date_time


def dict_to_csv(dict, csvname, mode, orient, reverse=False, header=True):
    if orient == 'index':
        df = pd.DataFrame.from_dict(dict, orient='index')
        df.to_csv(csvname, header=False, mode=mode)
    if orient == 'columns':
        df = pd.DataFrame(dict, index=[0])
        if reverse: #reverse dataframe columnes
            df = df.iloc[:, ::-1]
        df.to_csv(csvname, index=False, mode=mode, header=header)
    # TODO: append rows considering columns names


def check_changes_commited():
    message = subprocess.check_output(['git', 'status'])
    message_last_line = str(message.decode("utf-8")).split('\n')[-2]
    required_last_line = 'nothing to commit, working tree clean'
    if message_last_line == required_last_line:
        status = True
    else:
        status = False
    return status


def save_plot(record, key, n_eval, fname):
    y = [o[key] for o in record]
    x = np.array(range(len(y))) / n_eval
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    ax.set_xlabel('epoch')
    ax.set_ylabel(key)
    fig.savefig(fname)


def save_plots(records, keys, labels, n_eval):
    for k in keys:
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel('epoch')
        ax.set_ylabel(k)
        for r, l in zip(records, labels):
            y = [o[k] for o in r]
            x = np.array(range(len(y))) / n_eval
            ax.plot(x, y, label=l)
        plt.legend()
        fname = k + '.png'
        fig.savefig(fname)
        plt.close()


def copy_files(arg_list, dest_dir):
    for a in arg_list:
        for file in glob.glob(a):
            shutil.copy(file, dest_dir)
# functions for choosing tokenizer, optimizer and model


def choose_tokenizer(tokenizer):
    if tokenizer == 'whitespace':
        return WhitespaceTokenizer()
    elif tokenizer == 'custom':
        return CustomTokenizer()
    elif tokenizer == 'lowerspacy':
        return LowerSpacy()
    elif tokenizer == 'gnews_sw':
        return GNewsTokenizerSW()
    elif tokenizer == 'gnews_num':
        return GNewsTokenizerNum()
    else:
        return tokenizer


def choose_model(model_name, text, n_layers, hidden_dim, dropout):
    model = BiLSTMPool(text.vocab.vectors,
                       lstm_layer=n_layers,
                       padding_idx=text.vocab.stoi[text.pad_token],
                       hidden_dim=hidden_dim,
                       dropout=dropout).cuda()
    return model


def choose_optimizer(params, args):
    if args.optim == 'Adam':
        optimizer = optim.Adam(params, lr=args.lr)
    elif args.optim == 'AdamW':
        optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.99))
    return optimizer

def lower_spacy(x):
    spacy_en = spacy.load('en')
    tokens = [tok.text.lower() for tok in spacy_en.tokenizer(x)]
    return tokens


class LowerSpacy(object):
    def __init__(self):
        self.tokenizer = spacy.load('en').tokenizer

    def __call__(self, x):
        return [tok.text.lower() for tok in self.tokenizer(x)]


class WhitespaceTokenizer(object):
    def __init__(self):
        pass

    def __call__(self, text):
        words = text.split(' ')
        return words


class CustomTokenizer(object):
    def __init__(self):
        #self.allowed_re = re.compile('^[A-Za-z0-9.,?!()]*$')
        self.punct = re.compile('[,!.?()]')

    def __call__(self, text):
        substrings = text.split(' ')
        tokens = []
        for ss in substrings:
            punct_match = self.punct.search(ss)
            if punct_match:
                start = punct_match.start()
                ss_tokens = [ss[:start], ss[start], ss[start + 1:]]
                ss_tokens = [t for t in ss_tokens if len(t) > 0]
            else:
                ss_tokens = [ss]
            tokens.extend(ss_tokens)
        return tokens


class GNewsTokenizerSW(object):
    def __init__(self):
        self.spacy_en = spacy.load('en')
        self.tokenizer = self.spacy_en.tokenizer
        self.remove_all_stopwords()
        self.add_stopwords(GNEWS_STOP_WORDS)

    def __call__(self, x):
        result = [tok.text for tok in self.tokenizer(x) if not tok.is_stop]
        if len(result) == 0:
            return ['<zero_length>']
        else:
            return result

    def remove_all_stopwords(self):
        spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
        for w in spacy_stopwords:
            lexeme = self.spacy_en.vocab[w]
            lexeme.is_stop = False

    def add_stopwords(self, custom_stop_words):
        for w in custom_stop_words:
            lexeme = self.spacy_en.vocab[w]
            lexeme.is_stop = True


class GNewsTokenizerNum(object):
    def __init__(self):
        self.spacy_en = spacy.load('en')
        self.tokenizer = self.spacy_en.tokenizer
        self.num_symb_re = re.compile('^[^a-zA-Z]*[0-9]+[^a-zA-Z]*$')
        self.sub_re = re.compile('[0-9]')

    def tokenize_numbers(self, tok):
        if self.num_symb_re.match(tok) and len(tok) > 1:
            tok = self.sub_re.sub('#', tok)
        return tok

    def __call__(self, x):
        return [self.tokenize_numbers(tok.text) for tok in self.tokenizer(x)]



def section_sizes_and_lengths(lengths):
    bs = len(lengths)
    flags = lengths[1:] - lengths[:-1]
    cuts = torch.add(torch.nonzero(flags).view(-1), 1)
    cuts = torch.cat([torch.tensor([0]), cuts, torch.tensor([bs])])
    section_lengths = lengths[cuts[:-1]]
    sizes = cuts[1:] - cuts[:-1]
    return sizes, section_lengths


def max_packed(x, lengths):
    sizes, section_lengths = section_sizes_and_lengths(lengths)
    sizes = sizes.cpu().numpy().tolist()
    tensors = torch.split(x, split_size_or_sections=sizes, dim=1)
    tensors = [t[:sl] for t, sl in zip(tensors, section_lengths)]
    maxes = [torch.max(t, 0)[0] for t in tensors]
    max_tensor = torch.cat(maxes)
    return max_tensor


def mean_packed(x, lengths):
    sizes, section_lengths = section_sizes_and_lengths(lengths)
    sizes = sizes.cpu().numpy().tolist()
    tensors = torch.split(x, split_size_or_sections=sizes, dim=1)
    tensors = [t[:sl] for t, sl in zip(tensors, section_lengths)]
    means = [torch.mean(t, 0) for t in tensors]
    mean_tensor = torch.cat(means)
    return mean_tensor


def out_max_mean(x):
    out = x[-1]
    max_tensor, _ = torch.max(x, 0)
    mean_tensor = torch.mean(x, 0)
    return out, max_tensor, mean_tensor


def out_max_mean_packed(x, lengths):
    if lengths[0] == lengths[-1]:
        return out_max_mean(x)
    sizes, section_lengths = section_sizes_and_lengths(lengths)
    sizes = sizes.cpu().numpy().tolist()
    tensors = torch.split(x, split_size_or_sections=sizes, dim=1)
    tensors = [t[:sl] for t, sl in zip(tensors, section_lengths)]
    out = torch.cat([t[-1] for t in tensors])
    max_tensor = torch.cat([torch.max(t, 0)[0] for t in tensors])
    mean_tensor = torch.cat([torch.mean(t, 0) for t in tensors])
    return out, max_tensor, mean_tensor


class BiLSTM(nn.Module):
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=100, lstm_layer=2, dropout=0.2):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            dropout=dropout,
                            bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim * lstm_layer * 2, 1)
        self.cell = self.lstm

    def forward(self, sents, length):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        lstm_out, (h_n, c_n) = self.lstm(x)
        y = self.hidden2label(self.dropout(torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=1)))
        return y


class BiGRU(nn.Module):
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=100, lstm_layer=2, dropout=0.2):
        super(BiGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.gru = nn.GRU(input_size=self.embedding.embedding_dim,
                          hidden_size=hidden_dim,
                          num_layers=lstm_layer,
                          dropout=dropout,
                          bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim * lstm_layer * 2, 1)
        self.cell = self.gru

    def forward(self, sents, length):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        gru_out, h_n = self.gru(x)
        y = self.hidden2label(self.dropout(torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=1)))
        return y


class BiLSTMPoolOld(nn.Module):
    # constant length for all sequences in batch
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=100, lstm_layer=2, dropout=0.2):
        super(BiLSTMPoolOld, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            dropout=dropout,
                            bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim * 6, 1)
        self.cell = self.lstm

    def forward(self, sents, length):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        lstm_out, (h_n, c_n) = self.lstm(x)
        sl, bs, _ = lstm_out.shape
        lstm_out = lstm_out.view(sl, bs, 2 * self.hidden_dim)
        output = lstm_out[-1]
        max_pool, _ = torch.max(lstm_out, 0)
        average_pool = torch.mean(lstm_out, 0)
        y = self.hidden2label(self.dropout(torch.cat((max_pool, average_pool, output), dim=1)))
        return y



class BiLSTMPoolSlow(nn.Module):
    # variable length for sequences in batch
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=100, lstm_layer=2, dropout=0.2):
        super(BiLSTMPoolSlow, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            dropout=dropout,
                            bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim * 6, 1)
        self.cell = self.lstm

    def forward(self, sents, lengths):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        lengths = lengths.view(-1).tolist()
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths)
        lstm_out, (h_n, c_n) = self.lstm(packed_x)
        unpacked_out, unpacked_len = nn.utils.rnn.pad_packed_sequence(lstm_out)
        sl, bs, _ = unpacked_out.shape
        lstm_out = unpacked_out
        output_list = [lstm_out[:l, i, :] for i, l in enumerate(unpacked_len.cpu().numpy())]
        output = torch.stack([t[-1, :] for t in output_list], dim=1)
        max_pool= torch.stack([torch.max(t, 0)[0] for t in output_list], dim=1)
        average_pool = torch.stack([torch.mean(t, 0) for t in output_list], dim=1)
        long_output = torch.cat((output, max_pool, average_pool), dim=0)
        long_output = torch.transpose(long_output, dim0=1, dim1=0)
        y = self.hidden2label(self.dropout(long_output))
        return y


class BiLSTMPoolFast(nn.Module):
    # varibale length for sequences in batch,  optimized for performance
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=100, lstm_layer=2, dropout=0.2):
        super(BiLSTMPoolFast, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = lstm_layer
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            dropout=dropout,
                            bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim * 6, 1)
        self.cell = self.lstm


    def forward(self, sents, lengths=None):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths)
        lstm_out, (h_n, c_n) = self.lstm(packed_x)
        unpacked_out, unpacked_len = nn.utils.rnn.pad_packed_sequence(lstm_out)
        sl, bs, _ = unpacked_out.shape
        output = h_n.view(self.num_layers, 2, bs, self.hidden_dim)[1]
        output = torch.cat((output[0], output[1]), dim=1)
        max_pool, _ = torch.max(unpacked_out, 0)
        average_pool = torch.mean(unpacked_out, 0)
        long_output = torch.cat((output, max_pool, average_pool), dim=1)
        #long_output = torch.transpose(long_output, dim0=1, dim1=0)
        y = self.hidden2label(self.dropout(long_output))
        return y


class BiLSTMPool(nn.Module):
    # varibale length for sequences in batch,  optimized for performance
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=100, lstm_layer=2, dropout=0.2):
        super(BiLSTMPool, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = lstm_layer
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            dropout=dropout,
                            bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim * 6, 1)
        self.cell = self.lstm

    def forward(self, sents, lengths=None):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths)
        lstm_out, (h_n, c_n) = self.lstm(packed_x)
        unpacked_out, unpacked_len = nn.utils.rnn.pad_packed_sequence(lstm_out)
        output, max_pool, average_pool = out_max_mean_packed(unpacked_out, unpacked_len)
        long_output = torch.cat((output, max_pool, average_pool), dim=1)
        y = self.hidden2label(self.dropout(long_output))
        return y



class BiLSTM_2FC(nn.Module):
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=100, lstm_layer=2, dropout=0.2):
        super(BiLSTM_2FC, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            dropout=dropout,
                            bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * lstm_layer * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.cell = self.lstm

    def forward(self, sents, length):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        lstm_out, (h_n, c_n) = self.lstm(x)
        y = self.fc1(self.dropout(torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=1)))
        y = self.fc2(self.dropout(y))
        return y

class BiGRUPool(nn.Module):
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=100, lstm_layer=2, dropout=0.2):
        super(BiGRUPool, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.gru = nn.GRU(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            dropout=dropout,
                            bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim * 6, 1)
        self.cell = self.gru

    def forward(self, sents, length):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        lstm_out, _ = self.gru(x)
        sl, bs, _ = lstm_out.shape
        lstm_out = lstm_out.view(sl, bs, 2 * self.hidden_dim)
        output = lstm_out[-1]
        max_pool, _ = torch.max(lstm_out, 0)
        average_pool = torch.mean(lstm_out, 0)
        y = self.hidden2label(self.dropout(torch.cat((max_pool, average_pool, output), dim=1)))
        return y


class BiGRUPool_2FC(nn.Module):
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=100, lstm_layer=2, dropout=0.2):
        super(BiGRUPool_2FC, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.gru = nn.GRU(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            dropout=dropout,
                            bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 6, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.cell = self.gru

    def forward(self, sents, length):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        lstm_out, _ = self.gru(x)
        sl, bs, _ = lstm_out.shape
        lstm_out = lstm_out.view(sl, bs, 2 * self.hidden_dim)
        output = lstm_out[-1]
        max_pool, _ = torch.max(lstm_out, 0)
        average_pool = torch.mean(lstm_out, 0)
        y = self.fc1(self.dropout(torch.cat((max_pool, average_pool, output), dim=1)))
        y = self.fc2(self.dropout(y))
        return y

class BiLSTMPool_2FC(nn.Module):
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=100, lstm_layer=2, dropout=0.2):
        super(BiLSTMPool_2FC, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.gru = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            dropout=dropout,
                            bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 6, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.cell = self.lstm

    def forward(self, sents, length):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        lstm_out, _, _ = self.lstm(x)
        sl, bs, _ = lstm_out.shape
        lstm_out = lstm_out.view(sl, bs, 2 * self.hidden_dim)
        output = lstm_out[-1]
        max_pool, _ = torch.max(lstm_out, 0)
        average_pool = torch.mean(lstm_out, 0)
        y = self.fc1(self.dropout(torch.cat((max_pool, average_pool, output), dim=1)))
        y = self.fc2(self.dropout(y))
        return y# Overriding torchtext methods


class MyTabularDataset(TabularDataset):
    """Subclass of torch.data.dataset.TabularDataset for k-fold cross-validation"""

    def split(self, split_ratio=0.8, stratified=False, strata_field='label',
              random_state=None):
        splits = super().split(split_ratio, stratified, strata_field, random_state)
        return [splits]

    def split_kfold(self, k, stratified=False, is_test=False, random_state=None):
        def _iter_folds():
            i = 0
            while i < k:
                # val index
                val_start_idx = cut_idxs[i]
                val_end_idx = cut_idxs[i + 1]
                val_index = randperm[val_start_idx:val_end_idx]

                # test index
                if is_test:
                    if i <= k - 2:
                        test_start_idx = cut_idxs[i + 1]
                        test_end_idx = cut_idxs[i + 2]
                    else:
                        test_start_idx = cut_idxs[0]
                        test_end_idx = cut_idxs[1]
                    test_index = randperm[test_start_idx:test_end_idx]
                else:
                    test_index = []

                # train index
                train_index = list(set(randperm) - set(test_index) - set(val_index))

                # split examples by index and create datasets
                train_data, val_data, test_data = tuple([self.examples[idx] for idx in index]
                                                        for index in [train_index, val_index, test_index])
                splits = tuple(Dataset(d, self.fields)
                               for d in (train_data, val_data, test_data) if d)

                # In case the parent sort key isn't none
                if self.sort_key:
                    for subset in splits:
                        subset.sort_key = self.sort_key
                i += 1
                yield splits

        rnd = RandomShuffler(random_state)
        N = len(self.examples)
        randperm = rnd(range(N))
        fold_len = int(N/k)
        cut_idxs = [e * fold_len for e in list(range(k))] + [N]
        data_iter = _iter_folds()
        return data_iter


class MyVectors(Vectors):
    #def __init__(self, *args, **kwargs):
    #    self.tokens = 0
    #    super(MyVectors, self).__init__(*args, *kwargs)
    def __init__(self, name, cache=None, to_cache = True,
                 url=None, unk_init=None, max_vectors=None):
        """
        Arguments:
           name: name of the file that contains the vectors
           cache: directory for cached vectors
           url: url for download if vectors not found in cache
           unk_init (callback): by default, initialize out-of-vocabulary word vectors
               to zero vectors; can be any function that takes in a Tensor and
               returns a Tensor of the same size
           max_vectors (int): this can be used to limit the number of
               pre-trained vectors loaded.
               Most pre-trained vector sets are sorted
               in the descending order of word frequency.
               Thus, in situations where the entire set doesn't fit in memory,
               or is not needed for another reason, passing `max_vectors`
               can limit the size of the loaded set.
         """
        cache = '.vector_cache' if cache is None else cache
        self.itos = None
        self.stoi = None
        self.vectors = None
        self.dim = None
        self.unk_init = torch.Tensor.zero_ if unk_init is None else unk_init
        if to_cache:
            self.cache(name, cache, url=url, max_vectors=max_vectors)
        else:
            self.load(name, max_vectors=max_vectors)


    def __getitem__(self, token):
        if token in self.stoi:
            #self.tokens += 1
            #print(self.tokens, self.low_tokens)
            return self.vectors[self.stoi[token]]
        elif token.lower() in self.stoi:
            #self.low_tokens += 1
            return self.vectors[self.stoi[token.lower()]]
        else:
            return self.unk_init(torch.Tensor(self.dim))

    def cache(self, name, cache, url=None, max_vectors=None):
        if os.path.isfile(name):
            path = name
            if max_vectors:
                file_suffix = '_{}.pt'.format(max_vectors)
            else:
                file_suffix = '.pt'
            path_pt = os.path.join(cache, os.path.basename(name)) + file_suffix
        else:
            path = os.path.join(cache, name)
            if max_vectors:
                file_suffix = '_{}.pt'.format(max_vectors)
            else:
                file_suffix = '.pt'
            path_pt = path + file_suffix

        if not os.path.isfile(path_pt):
            if not os.path.isfile(path) and url:
                logger.info('Downloading vectors from {}'.format(url))
                if not os.path.exists(cache):
                    os.makedirs(cache)
                dest = os.path.join(cache, os.path.basename(url))
                if not os.path.isfile(dest):
                    with tqdm(unit='B', unit_scale=True, miniters=1, desc=dest) as t:
                        try:
                            urlretrieve(url, dest, reporthook=reporthook(t))
                        except KeyboardInterrupt as e:  # remove the partial zip file
                            os.remove(dest)
                            raise e
                logger.info('Extracting vectors into {}'.format(cache))
                ext = os.path.splitext(dest)[1][1:]
                if ext == 'zip':
                    with zipfile.ZipFile(dest, "r") as zf:
                        zf.extractall(cache)
                elif ext == 'gz':
                    if dest.endswith('.tar.gz'):
                        with tarfile.open(dest, 'r:gz') as tar:
                            tar.extractall(path=cache)
            if not os.path.isfile(path):
                raise RuntimeError('no vectors found at {}'.format(path))

            logger.info("Loading vectors from {}".format(path))

            itos, vectors, dim = read_emb(path, max_vectors)

            self.itos = itos
            self.stoi = {word: i for i, word in enumerate(itos)}
            self.vectors = torch.Tensor(vectors).view(-1, dim)
            self.dim = dim
            logger.info('Saving vectors to {}'.format(path_pt))
            if not os.path.exists(cache):
                os.makedirs(cache)
            torch.save((self.itos, self.stoi, self.vectors, self.dim), path_pt)
        else:
            logger.info('Loading vectors from {}'.format(path_pt))
            self.itos, self.stoi, self.vectors, self.dim = torch.load(path_pt)

    def load(self, path, max_vectors=None):
        print('Loading embedding vectors. No cache')
        if not os.path.isfile(path):
            raise RuntimeError('no vectors found at {}'.format(path))

        logger.info("Loading vectors from {}".format(path))

        itos, vectors, dim = read_emb(path, max_vectors)

        self.itos = itos
        self.stoi = {word: i for i, word in enumerate(itos)}
        self.vectors = torch.Tensor(vectors).view(-1, dim)
        self.dim = dim


def read_emb(path, max_vectors):
    ext = os.path.splitext(path)[1][1:]
    if ext == 'bin':
        itos, vectors, dim = emb_from_bin(path, max_vectors)
    else:
        itos, vectors, dim = emb_from_txt(path, ext, max_vectors)
    return itos, vectors, dim


def emb_from_txt(path, ext, max_vectors):
    if ext == 'gz':
        open_file = gzip.open
    else:
        open_file = open

    vectors_loaded = 0
    with open_file(path, 'rb') as f:
        num_lines, dim = _infer_shape(f)
        if not max_vectors or max_vectors > num_lines:
            max_vectors = num_lines

        itos, vectors, dim = [], torch.zeros((max_vectors, dim)), None

        for line in f:
            # get rid of Unicode non-breaking spaces in the vectors.
            entries = line.rstrip().split(b" ")

            word, entries = entries[0], entries[1:]
            if dim is None and len(entries) > 1:
                dim = len(entries)
            elif len(entries) == 1:
                logger.warning("Skipping token {} with 1-dimensional "
                               "vector {}; likely a header".format(word, entries))
                continue
            elif dim != len(entries):
                raise RuntimeError(
                    "Vector for token {} has {} dimensions, but previously "
                    "read vectors have {} dimensions. All vectors must have "
                    "the same number of dimensions.".format(word, len(entries),
                                                            dim))

            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except UnicodeDecodeError:
                logger.info("Skipping non-UTF8 token {}".format(repr(word)))
                continue

            vectors[vectors_loaded] = torch.tensor([float(x) for x in entries])
            vectors_loaded += 1
            itos.append(word)

            if vectors_loaded == max_vectors:
                break

    return itos, vectors, dim


def emb_from_bin(path, max_vectors):
    emb_index = KeyedVectors.load_word2vec_format(path, limit = max_vectors, binary=True)
    itos = emb_index.index2word
    vectors = emb_index.vectors
    dim = emb_index.vector_size
    return itos, vectors, dim

def _infer_shape(f):
    num_lines, vector_dim = 0, None
    for line in f:
        if vector_dim is None:
            row = line.rstrip().split(b" ")
            vector = row[1:]
            # Assuming word, [vector] format
            if len(vector) > 2:
                # The header present in some (w2v) formats contains two elements.
                vector_dim = len(vector)
                num_lines += 1  # First element read
        else:
            num_lines += 1
    f.seek(0)
    return num_lines, vector_dim


class Learner:

    def __init__(self, model, dataloaders, loss_func, optimizer, scheduler, args):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.recorder = Recorder(args)
        self.args = args

        if len(dataloaders) == 3:
            self.train_dl, self.val_dl, self.test_dl = dataloaders
        elif len(dataloaders) == 2:
            self.train_dl, self.val_dl = dataloaders
            self.test_dl = None
        elif len(dataloaders) == 1:
            self.train_dl = dataloaders
            self.val_dl = self.test_dl = None

    @staticmethod
    def to_cuda(data):
        if type(data) == tuple:
            return [tensor.cuda() for tensor in data]
        else:
            return [data.cuda(), None]

    def fit(self, epoch, n_eval, tresh, early_stop, warmup_epoch, clip):

        step = 0
        min_loss = 1e5
        max_f1 = 0
        max_test_f1 = 0
        no_improve_epoch = 0
        no_improve_in_previous_epoch = False
        fine_tuning = False
        losses = []
        best_test_info = None
        torch.backends.cudnn.benchmark = False
        eval_every = int(len(list(iter(self.train_dl))) / n_eval)

        time_start = time.time()
        for e in range(epoch):
            self.scheduler.step()
            if e >= warmup_epoch:
                if no_improve_in_previous_epoch:
                    no_improve_epoch += 1
                    if no_improve_epoch >= early_stop:
                        e = e-1
                        break
                else:
                    no_improve_epoch = 0
                no_improve_in_previous_epoch = True
            if not fine_tuning and e >= warmup_epoch:
                self.model.embedding.weight.requires_grad = True
                fine_tuning = True
            self.train_dl.init_epoch()

            for train_batch in iter(self.train_dl):
                step += 1
                self.model.zero_grad()
                self.model.train()
                model_input = self.to_cuda(train_batch.text)
                y = train_batch.target.type(torch.Tensor).cuda()
                pred = self.model.forward(*model_input).view(-1)
                loss = self.loss_func(pred, y)
                self.recorder.tr_record.append({'tr_loss': loss.cpu().data.numpy()})
                loss.backward()
                total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                # parameters = list(filter(lambda p: p.grad is not None, self.model.parameters()))
                # total_norm = 0
                # for p in parameters:
                #     param_norm = p.grad.data.norm(2)
                #     total_norm += param_norm.item() ** 2
                # total_norm = total_norm ** (1. / 2)

                self.recorder.norm_record.append({'grad_norm': total_norm})
                self.optimizer.step()

                if step % eval_every == 0:
                    with torch.no_grad():
                        # train evaluation
                        losses.append(loss.cpu().data.numpy())
                        train_loss = np.mean(losses)

                        # val evaluation
                        val_loss, val_f1 = self.evaluate(self.val_dl, tresh)
                        self.recorder.val_record.append({'step': step, 'loss': val_loss, 'f1': val_f1})
                        info = {'best_ep': e, 'step': step, 'train_loss': train_loss,
                                'val_loss': val_loss, 'val_f1': val_f1}
                        self.recorder.save_step(info, message=True)
                        if val_f1  > max_f1:
                            self.recorder.save(self.model, info)
                            max_f1 = val_f1
                            no_improve_in_previous_epoch = False
                        if val_loss < min_loss:
                            min_loss = val_loss

                        # test evaluation
                        # if self.args.test:
                        #     test_loss, test_f1 =  self.evaluate(self.test_dl, tresh)
                        #    test_info = {'test_ep': e, 'test_step': step, 'test_loss': test_loss, 'test_f1': test_f1}
                        #    self.recorder.test_record.append({'step': step, 'loss': test_loss, 'f1': test_f1})
                        #     print('epoch {:02} - step {:06} - test_loss {:.4f} - test_f1 {:.4f}'.format(*list(test_info.values())))
                         #   if test_f1 >= max_test_f1:
                         #       max_test_f1 = test_f1
                         #       best_test_info = test_info

        tr_time = print_duration(time_start, 'training time: ')
        self.recorder.append_info({'ep_time': tr_time/(e + 1)})

        #if self.args.test:
        #    self.recorder.append_info(best_test_info, message='Best results for test:')
        self.recorder.append_info({'min_loss': min_loss}, 'min val loss: ')

        self.model, info = self.recorder.load(message = 'best model:')

        # final train evaluation
        train_loss, train_f1 = self.evaluate(self.train_dl, tresh)
        tr_info = {'train_loss':train_loss, 'train_f1':train_f1}
        self.recorder.append_info(tr_info, message='train loss and f1:')


    def evaluate(self, dl, tresh):
        with torch.no_grad():
            dl.init_epoch()
            self.model.cell.flatten_parameters()
            self.model.eval()
            self.model.zero_grad()
            loss = []
            tp = 0
            n_targs = 0
            n_preds = 0
            for batch in iter(dl):
                model_input = self.to_cuda(batch.text)
                y = batch.target.type(torch.Tensor).cuda()
                pred = self.model.forward(*model_input).view(-1)
                loss.append(self.loss_func(pred, y).cpu().data.numpy())
                label = (torch.sigmoid(pred).cpu().data.numpy() > tresh).astype(int)
                y = y.cpu().data.numpy()
                tp += sum(y + label == 2)
                n_targs += sum(y)
                n_preds += sum(label)
            f1 = f1_metric(tp, n_targs, n_preds)
            loss = np.mean(loss)
        return loss, f1

    def predict_probs(self, is_test=False):
        if is_test:
            print('predicting test dataset...')
            dl = self.test_dl
        else:
            print('predicting validation dataset...')
            dl = self.val_dl

        self.model.cell.flatten_parameters()
        self.model.eval()
        y_pred = []
        y_true = []
        ids = []

        dl.init_epoch()
        for batch in iter(dl):
            model_input = self.to_cuda(batch.text)
            if not is_test or self.args.test:
                y_true += batch.target.data.numpy().tolist()
            y_pred += torch.sigmoid(self.model.forward(*model_input).view(-1)).cpu().data.numpy().tolist()
            ids += batch.qid.view(-1).data.numpy().tolist()
        return y_pred, y_true, ids

    def predict_labels(self, is_test=False, thresh=0.5):
        y_prob, y_true, ids = self.predict_probs(is_test=is_test)

        if type(thresh) == list:
            thresh, max_f1 = choose_thresh(y_prob, y_true, *thresh, message=True)
            self.recorder.append_info({'best_tr': thresh, 'best_f1': max_f1})

        y_label = (np.array(y_prob) >= thresh).astype(int)
        return y_label, y_prob, y_true, ids, thresh





class Recorder:

    models_dir = './models'
    best_model_path = './models/best_model.m'
    best_info_path = './models/best_model.info'
    record_dir = './notes'

    def __init__(self, args):
        self.args = args
        if self.args.mode == 'test':
            self.record_path = './notes/test_records.csv'
        elif self.args.mode == 'run':
            self.record_path = './notes/records.csv'
        self.val_record = []
        self.tr_record = []
        self.test_record = []
        self.norm_record = []
        self.new = True

    @classmethod
    def append_info(cls, dict, message=None):
        dict = format_info(dict)
        if message:
            print(message, dict)
        info = torch.load(cls.best_info_path)
        info.update(dict)
        torch.save(info, cls.best_info_path)

    def save(self, model, info):
        os.makedirs(self.models_dir, exist_ok=True)
        torch.save(model, self.best_model_path)
        info = format_info(info)
        torch.save(info, self.best_info_path)

    def load(self, message=None):
        model = torch.load(self.best_model_path)
        info = torch.load(self.best_info_path)
        if message:
            print(message, info)
        return model, info

    def save_step(self, step_info, message = False):
        if self.new:
            header = True
            mode = 'w'
            self.new = False
        else:
            header=False
            mode = 'a'
        dict_to_csv(step_info, 'train_steps.csv', mode, orient='columns', header=header)
        if message:
            print('epoch {:02} - step {:06} - train_loss {:.4f} - val_loss {:.4f} - f1 {:.4f}'.format(
                *list(step_info.values())))

    def record(self, fold):
        # save plots
        save_plot(self.val_record, 'loss', self.args.n_eval, 'val_loss')
        save_plot(self.val_record, 'f1', self.args.n_eval, 'val_f1')
        save_plot(self.norm_record, 'grad_norm', self.args.n_eval, 'grad_norm')
        if self.args.test:
            save_plots([self.val_record, self.test_record], ['loss', 'f1'], ['val', 'test'],self.args.n_eval)

        # create subdir for this experiment
        os.makedirs(self.record_dir, exist_ok=True)
        subdir = os.path.join(self.models_dir, str_date_time())
        if self.args.mode == 'test':
            subdir +=  '_test'
        os.mkdir(subdir)

        # write model params and results to csv
        csvlog = os.path.join(subdir, 'info.csv')
        param_dict = {}
        for arg in vars(self.args):
            param_dict[arg] = str(getattr(self.args, arg))
        info = torch.load(self.best_info_path)
        hash = get_hash() if self.args.machine == 'dt' else 'no_hash'
        passed_args = ' '.join(sys.argv[1:])
        param_dict = {'hash':hash, 'subdir':subdir, **param_dict, **info, 'args': passed_args}
        dict_to_csv(param_dict, csvlog, 'w', 'index', reverse=False)
        header = True if fold == 0 else False
        dict_to_csv(param_dict, self.record_path, 'a', 'columns', reverse=True, header=header)

        # copy all records to subdir
        png_files = ['val_loss.png', 'val_f1.png'] if not self.args.test else ['loss.png', 'f1.png']
        csv_files = ['val_probs.csv', 'train_steps.csv', 'submission.csv', 'test_probs.csv']
        copy_files([*png_files, 'models/*.info', *csv_files], subdir)
        return subdir


def format_info(info):
    keys = list(info.keys())
    values = list(info.values())
    for k, v in zip(keys, values):
        info[k] = round(v, 4)
    return info


def choose_thresh(probs, true, thresh_range, message=True):
    min_th, max_th, th_step = thresh_range
    tmp = [0, 0, 0]  # idx, current_f1, max_f1
    th = min_th
    for tmp[0] in np.arange(min_th, max_th, th_step):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tmp[1] = f1_score(true, probs > tmp[0])
        if tmp[1] > tmp[2]:
            th = tmp[0]
            tmp[2] = tmp[1]
    if message:
        print('best threshold is {:.4f} with F1 score: {:.4f}'.format(th, tmp[2]))

    return th, tmp[2]#!/usr/bin/env python





def parse_main_args(main_args=None):
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--machine', default='dt', choices=['dt', 'kaggle'])
    arg('--mode', default='run', choices=['test', 'run'])

    # data preprocessing params
    arg('--kfold', '-k', type=int)
    arg('--split_ratio', '-sr', nargs='+', default=[0.8], type=float)
    arg('--test', action='store_true') # if present split data in train-val-test else split train-val
    arg('--seed', default=2018, type=int)
    arg('--tokenizer', '-t', default='spacy', choices=['spacy', 'whitespace', 'custom', 'lowerspacy', 'gnews_sw', 'gnews_num'])
    arg('--embedding', '-em', default='glove', choices=['glove', 'gnews', 'paragram', 'wnews'])
    arg('--max_vectors', '-mv', default=5000000, type=int)
    arg('--no_cache', action='store_true')
    arg('--var_length', '-vl', action = 'store_false') # variable sequence length in batches
    arg('--unk_std', '-us', default = 0.001, type=float)

    # training params
    arg('--optim', '-o', default='Adam', choices=['Adam', 'AdamW'])
    arg('--epoch', '-e', default=7, type=int)
    arg('--lr', '-lr', default=1e-3, type=float)
    arg('--lrstep', default=[3], nargs='+', type=int) # steps when lr multiplied by 0.1
    arg('--batch_size', '-bs', default=512, type=int)
    arg('--n_eval', '-ne', default=1, type=int, help='Number of validation set evaluations during 1 epoch')
    arg('--warmup_epoch', '-we', default=2, type=int, help='Number of epochs without fine tuning')
    arg('--early_stop', '-es', default=2, type=int, help='Stop training if no improvement during this number of epochs')
    arg('--f1_tresh', '-ft', default=0.335, type=float)
    arg('--clip', type=float, default=1, help='gradient clipping')

    # model params
    arg('--model', '-m', default='BiLSTMPool', choices=['BiLSTM', 'BiGRU', 'BiLSTMPool', 'BiLSTM_2FC', 'BiGRUPool',
                                                    'BiGRUPool_2FC', 'BiLSTMPool_2FC', 'BiLSTMPoolFast', 'BiLSTMPoolOld',
                                                    'BiLSTMPoolTest'])
    arg('--n_layers', '-n', default=2, type=int, help='Number of layers in model')
    arg('--hidden_dim', '-hd', type=int, default=100)
    arg('--dropout', '-d', type=float, default=0.2)

    if main_args:
        args = parser.parse_args(main_args)
    else:
        args = parser.parse_args()
    return args


def analyze_args(args):
    if args.embedding == 'glove':
        emb_path = 'glove.840B.300d/glove.840B.300d.txt'
    elif args.embedding == 'paragram':
        emb_path = 'paragram_300_sl999/paragram_300_sl999.txt'
    elif args.embedding == 'gnews':
        emb_path = 'GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
    elif args.embedding == 'wnews':
        emb_path = 'wiki-news-300d-1M/wiki-news-300d-1M.vec'
    # TODO: add other embeddings

    if args.machine == 'dt':
        data_dir = cache = './data'
        if args.mode == 'run':
            if not check_changes_commited():
                sys.exit("Please commit all changes!")
    elif args.machine == 'kaggle':
        data_dir = '../input'
        cache = '.'
        args.no_cache = True

    train_csv = os.path.join(data_dir, 'train.csv')
    test_csv = os.path.join(data_dir, 'test.csv')
    emb_path = os.path.join(data_dir, 'embeddings', emb_path)

    if args.mode == 'test':
        # create smaller files for testing main function
        n_cut = 1000
        args.max_vectors = 10000
        args.batch_size = n_cut/100

        if args.machine == 'kaggle':
            data_dir = '.'

        train_small_csv, test_small_csv = reduce_datasets([train_csv, test_csv], data_dir, n_cut)
        train_csv, test_csv = train_small_csv, test_small_csv
    if args.mode == 'run':
        pass

    # split ratio should be float if len == 1
    sr = args.split_ratio
    if len(sr) == 1:
        args.split_ratio = sr[0]
    if len(sr) == 3:
        args.test = True
    return train_csv, test_csv, emb_path, cache


def job(args, train_csv, test_csv, embedding, cache):
    """ Main function. Reads data, makes preprocessing, trains model and records results.
        Gets args as argument and passes values of it's fields to functions."""

    data = Data(train_csv, test_csv, cache)

    # read and preprocess data
    data.preprocess(args.tokenizer, args.var_length)
    to_cache = not args.no_cache
    data.embedding_lookup(embedding, args.unk_std, args.max_vectors, to_cache)

    # split train dataset
    data_iter = data.split(args.kfold, args.split_ratio, args.test, args.seed)

    # iterate through folds
    loss_function = nn.BCEWithLogitsLoss()
    for fold, d in enumerate(data_iter):
        print(f'\n__________ fold {fold} __________')
        # get dataloaders
        if len(d) == 2:
            train, val = d
            test = data.test
        else:
            train, val, test = d
        dataloaders = iterate(train, val, test, args.batch_size) # train, val and test dataloader

        # choose model, optimizer, lr scheduler
        model = choose_model(args.model, data.text, args.n_layers, args.hidden_dim, args.dropout)
        optimizer = choose_optimizer(filter(lambda p: p.requires_grad, model.parameters()), args)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lrstep, gamma=0.1)
        learn = Learner(model, dataloaders, loss_function, optimizer, scheduler, args)
        learn.fit(args.epoch, args.n_eval, args.f1_tresh, args.early_stop, args.warmup_epoch, args.clip)

        # load best model
        learn.model, info = learn.recorder.load()
        # save val predictions
        y_pred, y_true, ids = learn.predict_probs()
        val_ids = [data.qid.vocab.itos[i] for i in ids]
        val_pred_to_csv(val_ids, y_pred, y_true)
        # choose best threshold for val predictions
        best_th, max_f1 = choose_thresh(y_pred, y_true, [0.1, 0.5, 0.01], message=True)
        learn.recorder.append_info({'best_th': best_th, 'max_f1': max_f1})


        # predict test labels
        test_label, test_prob,_, test_ids, tresh = learn.predict_labels(is_test=True, thresh=args.f1_tresh)
        if args.test:
            test_loss, test_f1 = learn.evaluate(learn.test_dl, args.f1_tresh)
            learn.recorder.append_info({'test_loss': test_loss, 'test_f1': test_f1}, message='Test set results: ')

        # save test predictions to submission.csv
        test_ids = [data.qid.vocab.itos[i] for i in test_ids]
        submit(test_ids, test_label, test_prob)
        record_path = learn.recorder.record(fold)  # directory path with all records
        print('\n')
    return record_path

def main(main_args=None):
    args = parse_main_args(main_args)
    train_csv, test_csv, emb_path, cache = analyze_args(args)
    record_path = job(args, train_csv, test_csv, emb_path, cache)
    return record_path



class Ensemble:

    ens_record_path = 'notes/ensemble.csv'

    def __init__(self, models, is_kfold=False, model_args='names'):
        self.val_pred_paths = [get_pred_path(m, 'val_probs.csv', model_args) for m in models]
        self.test_pred_paths = [get_pred_path(m, 'test_probs.csv', model_args) for m in models]
        self.pred_dirs = [get_pred_dir(m, model_args) for m in models]

    def __call__(self, args):
        thresh = args.thresh
        method = args.method

        # find best method parameters based on validation data
        y_preds = []
        last_ids = None
        for pp in self.val_pred_paths:
            ids, y_prob, y_true = load_pred_from_csv(pp)
            y_preds.append(y_prob)
            if last_ids:
                if last_ids != ids:
                    raise Exception('Prediction ids should be the same for ensemble')
        val_ens_prob = methods[method](y_preds, args) # target probability after ensembling
        thresh, max_f1 = self.evaluate_ensemble(val_ens_prob, y_true, thresh)
        self.record(max_f1, thresh, method)

        # predict test labels and save submission
        y_preds = []
        last_ids = None
        for pp in self.test_pred_paths:
            ids, y_prob = load_pred_from_csv(pp, is_label=False)
            y_preds.append(y_prob)
            if last_ids:
                if last_ids != ids:
                    raise Exception('Prediction ids should be the same for ensemble')
        test_ens_prob = methods[method](y_preds, args)  # target probability after ensembling
        test_ens_label = (test_ens_prob > thresh).astype(int)
        submit(ids, test_ens_label, test_ens_prob)


    @staticmethod
    def evaluate_ensemble(final_pred, true, thresh):
        if type(thresh) == float:
            f1 = f1_score(true, final_pred > thresh)
            return thresh, f1
        elif type(thresh) == list:
            best_thr, max_f1 = choose_thresh(final_pred, true, thresh)
            print('Best threshold for ensemble prediction is {:.4f} with F1 score: {:.4f}'.format(best_thr, max_f1))
            return best_thr, max_f1
        else:
            raise Exception('Threshold must be float or list of 3 floats')

    @staticmethod
    def read_model_info(path):
        descr = [[], []]
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                descr[0].append(row[0])
                descr[1].append(row[1])
        return descr


    def record(self, max_f1, tresh, method):
        ens_info = format_info({'max_f1': max_f1, 'tresh': tresh})
        ens_info = {'method': method, **ens_info}
        model_infos = [] # partial model descriptions
        # copy partial models descriptions
        info_paths = [os.path.join(pp, 'info.csv') for pp in self.pred_dirs]
        for ip in info_paths:
            info = self.read_model_info(ip)
            model_infos.append(info)
        model_infos = [o for l in model_infos for o in l]

        with open(self.ens_record_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerows(model_infos)
        dict_to_csv(ens_info, self.ens_record_path, 'a', 'columns', reverse=False, header=True)


def val_pred_to_csv(ids, y_pred, y_true, fname='val_probs.csv'):
    df = pd.DataFrame()
    df['qid'] = ids
    df['prediction'] = y_pred
    df['true_label'] = y_true
    df.to_csv(fname, index=False)


def get_pred_path(m, pred_file_name, model_args='names'):
    if model_args == 'names':
        path = os.path.join('./models', model_dict[m][0], pred_file_name)
    elif model_args == 'paths':
        path = os.path.join(m, pred_file_name)
    return path


def get_pred_dir(m, model_args='names'):
    if model_args == 'names':
        path = os.path.join('./models', model_dict[m][0])
    elif model_args == 'paths':
        path = m
    return path


def load_pred_from_csv(pred_path, is_label=True):
    df = pd.read_csv(pred_path)
    qid = df['qid']
    probs = df['prediction']
    true = df['true_label'] if is_label else None
    column_values = [d.values for d in [qid, probs, true] if d is not None]
    return column_values


def ens_parser(add_help=True):
    parser = argparse.ArgumentParser(add_help=add_help)
    arg = parser.add_argument
    arg('--models', '-m', nargs='+', type=str, default=['glove', 'wnews', 'paragram'])
    arg('-k', action='store_true')
    arg('--method', '-mth', default='mean', type=str, choices=['mean', 'weight', 'stack'])
    arg('--weights', '-w', nargs='+', default=[0.9, 0.1], type=float)
    arg('--thresh', '-th', nargs='+', default=[0.1, 0.5, 0.01], type=float)
    return parser


def parse_ens_args():
    parser = ens_parser()
    args = parser.parse_args()
    return args


#!/usr/bin/env python


def parse_ens_main_args():
    parser = argparse.ArgumentParser(parents=[ens_parser(add_help=False)])
    arg = parser.add_argument
    arg('--main_args', '-a', nargs='+', default=["-es 3 -e 8 -em wnews -hd 150 -we 10 --lrstep 10 -us 0.1", "-e 5 -hd 150 -us 0.1", "-e 5 -hd 150 -em paragram -us 0", "-em gnews -es 2 -e 8 -hd 150 -us 0.1"], type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_ens_main_args()
    main_args = [a.split() for a in args.main_args]
    record_dirs = []
    for a in main_args:
        record_dir = main(a)
        record_dirs.append(record_dir)
    ens = Ensemble(record_dirs, args.k, model_args='paths')
    ens(args)

# '--mode test -em glove', '--mode test -em wnews', '--mode test -em paragram', '--mode test -em gnews'
# "-es 3 -e 8 -em wnews -hd 150 -we 10 --lrstep 10 -us 0.1", "-e 5 -hd 150 -us 0.1", "-e 5 -hd 150 -em paragram -us 0", "-em gnews -es 2 -e 8 -hd 150 -us 0.1"