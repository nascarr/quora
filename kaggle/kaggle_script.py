#!/usr/bin/env python
import os
import pandas as pd
import subprocess
import pandas as pd
import numpy as np
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import glob
import shutil
from torchtext.data.dataset import *
import numpy as np
import torchtext.data as data
import torchtext.vocab as vocab
import time
import torch
import torch.nn as nn
import numpy as np
import torch
import os
from sklearn.metrics import f1_score
import warnings
import shutil
import time
import subprocess
import sys
import os
import sys
import random
import torch.nn as nn
import torch.optim as optim
import argparse

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


if __name__ == '__main__':
    directory = '../data'
    train_csv = os.path.join(directory, 'train.csv')
    test_csv = os.path.join(directory, 'test.csv')
    n = 1000
    n_emb = 10000
    emb_dir = os.path.join(directory, 'embeddings')
    emb_path = 'glove.840B.300d/glove.840B.300d.txt'

    reduce_datasets([train_csv, test_csv], n)
    reduce_embedding(emb_path, n_emb)

def submit(test_ids, prediciton, subm_name='submission.csv'):
    sub_df = pd.DataFrame()
    sub_df['qid'] = test_ids
    sub_df['prediction'] = prediciton
    sub_df.to_csv(subm_name, index=False)

    print('Predictions saved in submission.csv file')


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


def get_hash():
    hash = subprocess.check_output(['git', 'describe', '--always'])
    hash = hash.decode("utf-8")[1:-1]
    return hash


def str_date_time():
    struct_time = time.localtime()
    date_time = time.strftime('%b_%d_%Y__%H:%M:%S', struct_time)
    return date_time


def dict_to_csv(dict, csvname, mode, orient, reverse):
    if orient == 'index':
        df = pd.DataFrame.from_dict(dict, orient='index')
        df.to_csv(csvname, header=False, mode=mode)
    if orient == 'columns':
        df = pd.DataFrame(dict, index=[0])
        if reverse: #reverse dataframe columnes
            df = df.iloc[:, ::-1]
        df.to_csv(csvname, index=False, mode=mode)
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


def save_plot(record, key,n_eval):
    y = [o[key] for o in record]
    x = np.array(range(len(y))) / n_eval
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    ax.set_xlabel('epoch')
    ax.set_ylabel(key)
    fname = key + '.png'
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
                val_start_idx = cut_idxs[(i+1)%k]
                val_end_idx = cut_idxs[(i + 2)%k]
                val_index = randperm[val_start_idx:val_end_idx]

                # test index
                if is_test:
                    test_start_idx = cut_idxs[i]
                    test_end_idx = cut_idxs[i + 1]
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



def preprocess(train_csv, test_csv, tokenizer, embeddings, cache):
        # types of csv columns
        location = './cachedir'
        time_start = time.time()
        text = data.Field(batch_first=True, tokenize=tokenizer)
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
        print_duration(time_start, 'Time to read data?')
        text.build_vocab(train, test, min_freq=1)
        qid.build_vocab(train, test)
        print_duration(time_start, 'Time to read and tokenize data: ')

        # embeddings lookup
        print('Embedding lookup...')
        time_start = time.time()
        text.vocab.load_vectors(vocab.Vectors(embeddings, cache=cache))
        print_duration(time_start, 'Time for embedding lookup: ')

        return train, test, text, qid


def iterate(train, val, test, batch_size):
        train_iter = data.BucketIterator(dataset=train,
                                         batch_size=batch_size,
                                         sort_key=lambda x: x.text.__len__(),
                                         shuffle=True,
                                         sort=False)

        val_iter = data.BucketIterator(dataset=val,
                                       batch_size=batch_size,
                                       sort_key=lambda x: x.text.__len__(),
                                       train=False,
                                       sort=False)

        test_iter = data.BucketIterator(dataset=test,
                                        batch_size=batch_size,
                                        sort_key=lambda x: x.text.__len__(),
                                        sort=False,
                                        train=False)
        return train_iter, val_iter, test_iter



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

    def forward(self, sents):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        lstm_out, (h_n, c_n) = self.lstm(x)
        y = self.hidden2label(self.dropout(torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=1)))
        return y

class Learner:

    models_dir = './models'
    best_model_path = './models/best_model.m'
    best_info_path = './models/best_model.info'
    record_dir = './notes'

    def __init__(self, model, dataloaders, loss_func, optimizer, scheduler, args):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        if len(dataloaders) == 3:
            self.train_dl, self.val_dl, self.test_dl = dataloaders
        elif len(dataloaders) == 2:
            self.train_dl, self.val_dl = dataloaders
            self.test_dl = None
        elif len(dataloaders) == 1:
            self.train_dl = dataloaders
            self.val_dl = self.test_dl = None
        self.args = args
        self.val_record = []
        self.train_record = []
        self.test_record = []
        if self.args.mode == 'test':
            self.record_path = './notes/test_records.csv'
        elif self.args.mode == 'run':
            self.record_path = './notes/records.csv'

    def fit(self, epoch, eval_every, tresh, early_stop=1, warmup_epoch=2):
        print('Start training!')
        time_start = time.time()
        step = 0
        max_f1 = 0
        max_test_f1 = 0
        no_improve_epoch = 0
        no_improve_in_previous_epoch = False
        fine_tuning = False
        losses = []
        torch.backends.cudnn.benchmark = True
        best_test_info = None

        for e in range(epoch):
            self.scheduler.step()
            if e >= warmup_epoch:
                if no_improve_in_previous_epoch:
                    no_improve_epoch += 1
                    if no_improve_epoch >= early_stop:
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
                x = train_batch.text.cuda()
                y = train_batch.target.type(torch.Tensor).cuda()
                pred = self.model.forward(x).view(-1)
                loss = self.loss_func(pred, y)
                self.train_record.append({'tr_loss': loss.cpu().data.numpy()})
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    losses.append(loss.cpu().data.numpy())
                    train_loss = np.mean(losses)
                    if step % eval_every == 0:
                        val_loss, val_f1 = self.evaluate(self.val_dl, tresh)
                        self.val_record.append({'step': step, 'loss': val_loss, 'f1': val_f1})
                        info = self.format_info(info = {'best_ep': e, 'step': step, 'train_loss': train_loss,
                                'val_loss': val_loss, 'val_f1': val_f1})
                        print('epoch {:02} - step {:06} - train_loss {:.4f} - val_loss {:.4f} - f1 {:.4f}'.format(
                            *list(info.values())))

                        if val_f1  >= max_f1:
                            self.save(info)
                            max_f1 = val_f1
                            no_improve_in_previous_epoch = False

                        """if 'target' in next(iter(self.test_dl)).fields:
                            test_loss, test_f1 =  self.evaluate(self.test_dl, tresh)
                            test_info = {'test_ep': e, 'test_step': step, 'test_loss': test_loss, 'test_f1': test_f1}
                            self.test_record.append({'step': step, 'loss': test_loss, 'f1': test_f1})
                            print('epoch {:02} - step {:06} - test_loss {:.4f} - test_f1 {:.4f}'.format(*list(test_info.values())))
                            if test_f1 >= max_test_f1:
                                max_test_f1 = test_f1
                                best_test_info = test_info"""

        """"if best_test_info:
            self.append_info(best_test_info)
            print('Best results for test:', best_test_info)"""

        print_duration(time_start, 'Training time: ')

        # calculate train loss and train f1_score
        print('Evaluating model on train dataset')
        train_loss, train_f1 = self.evaluate(self.train_dl, tresh)
        tr_info = self.format_info({'train_loss':train_loss, 'train_f1':train_f1})
        print(tr_info)
        self.append_info(tr_info)

        m_info = self.load()
        print(f'Best model: {m_info}')

    def evaluate(self, dl, tresh):
        with torch.no_grad():
            dl.init_epoch()
            self.model.eval()
            self.model.zero_grad()
            loss = []
            tp = 0
            n_targs = 0
            n_preds = 0
            for batch in iter(dl):
                x = batch.text.cuda()
                y = batch.target.type(torch.Tensor).cuda()
                pred = self.model.forward(x).view(-1)
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
            print('Predicting test dataset...')
            dl = self.test_dl
        else:
            print('Predicting validation dataset...')
            dl = self.val_dl

        self.model.lstm.flatten_parameters()
        self.model.eval()
        y_pred = []
        y_true = []
        ids = []

        dl.init_epoch()
        for batch in iter(dl):
            x = batch.text.cuda()
            if not is_test:
                y_true += batch.target.data.numpy().tolist()
            y_pred += torch.sigmoid(self.model.forward(x).view(-1)).cpu().data.numpy().tolist()
            ids += batch.qid.view(-1).data.numpy().tolist()
        return y_pred, y_true, ids

    def predict_labels(self, is_test=False, tresh=0.5):
        def _choose_tr(self, min_tr, max_tr, tr_step):
            print('Choosing treshold.\n')
            val_pred, val_true, _ = self.predict_probs(is_test=False)
            tmp = [0, 0, 0]  # idx, current_f1, max_f1
            tr = min_tr
            for tmp[0] in np.arange(min_tr, max_tr, tr_step):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    tmp[1] = f1_score(val_true, np.array(val_pred) > tmp[0])
                if tmp[1] > tmp[2]:
                    tr = tmp[0]
                    tmp[2] = tmp[1]
            print('Best threshold is {:.4f} with F1 score: {:.4f}'.format(tr, tmp[2]))
            return tr, tmp[2]

        y_pred, y_true, ids = self.predict_probs(is_test=is_test)

        if type(tresh) == list:
            tresh, max_f1 = _choose_tr(self, *tresh)
            self.append_info({'best_tr': tresh, 'best_f1': max_f1})

        y_label = (np.array(y_pred) >= tresh).astype(int)
        return y_label, y_true, ids, tresh

    def save(self, info):
        os.makedirs(self.models_dir, exist_ok=True)
        if self.args.mode == 'run':
            torch.save(self.model, self.best_model_path)
        torch.save(info, self.best_info_path)

    @staticmethod
    def format_info(info):
        keys = list(info.keys())
        values = list(info.values())
        for k, v in zip(keys, values):
            info[k] = round(v, 4)
        return info

    @classmethod
    def append_info(cls, dict):
        dict = cls.format_info(dict)
        info = torch.load(cls.best_info_path)
        info.update(dict)
        torch.save(info, cls.best_info_path)

    def record(self):
        # save plots
        save_plot(self.val_record, 'loss', self.args.n_eval)
        save_plot(self.val_record, 'f1', self.args.n_eval)
        save_plot(self.train_record, 'tr_loss', self.args.n_eval)
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
        hash = get_hash()
        passed_args = ' '.join(sys.argv[1:])
        param_dict = {'hash':hash, 'subdir':subdir, **param_dict, **info, 'args': passed_args}
        dict_to_csv(param_dict, csvlog, 'w', 'index', reverse=False)
        dict_to_csv(param_dict, self.record_path, 'a', 'columns', reverse=True)

        # copy all records to subdir
        copy_files(['*.png', 'models/*.m', 'models/*.info'], subdir)

    def load(self):
        self.model = torch.load(self.best_model_path)
        info = torch.load(self.best_info_path)
        return info
#!/usr/bin/env python




def parse_script_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--machine', default='dt', choices=['dt', 'kaggle'])
    arg('--mode', default='run', choices=['test', 'run'])

    # data preprocessing params
    arg('--kfold', '-k', type=int)
    arg('--split_ratio', '-sr', nargs='+', default=0.8, type=float)
    arg('--seed', default=2018, type=int)
    arg('--tokenizer', '-t', default='spacy', choices=['spacy'])
    arg('--embedding', '-em', default='glove', choices=['glove', 'google_news', 'paragram', 'wiki_news'])

    # training params
    arg('--optim', '-o', default='Adam', choices=['Adam', 'AdamW'])
    arg('--epoch', '-e', default=7, type=int)
    arg('--lr', '-lr', default=1e-3, type=float)
    arg('--batch_size', '-bs', default=512, type=int)
    arg('--n_eval', '-ne', default=1, type=int, help='Number of validation set evaluations during 1 epoch')
    arg('--warmup_epoch', '-we', default=2, type=int, help='Number of epochs without fine tuning')
    arg('--early_stop', '-es', default=1, type=int, help='Stop training if no improvement during this number of epochs')
    arg('--f1_tresh', '-ft', default=0.33, type=float)

    # model params
    arg('--model', '-m', default='BiLSTM', choices=['BiLSTM'])
    arg('--n_layers', '-n', default=2, type=int, help='Number of layers in model')
    arg('--hidden_dim', '-hd', type=int, default=100)
    arg('--dropout', '-d', type=float, default=0.2)

    args = parser.parse_args()
    print(args)
    return args


def analyze_args(args):
    if args.embedding == 'glove':
        emb_path = 'embeddings/glove.840B.300d/glove.840B.300d.txt'
    # TODO: add other embeddings

    if args.machine == 'dt':
        data_dir = cache = './data'
    elif args.machine == 'kaggle':
        data_dir = '../input'
        cache = '.'

    train_csv = os.path.join(data_dir, 'train.csv')
    test_csv = os.path.join(data_dir, 'test.csv')
    emb_path = os.path.join(data_dir, emb_path)

    if args.mode == 'test':
        # create smaller files for testing main function
        n_cut = 1000
        n_cut_emb = 10000
        args.batch_size = n_cut/100

        if args.machine == 'kaggle':
            data_dir = '.'

        train_small_csv, test_small_csv = reduce_datasets([train_csv, test_csv], data_dir, n_cut)
        emb_small_path = reduce_embedding(emb_path, data_dir, n_cut_emb)
        train_csv, test_csv, emb_path = train_small_csv, test_small_csv, emb_small_path
    elif args.mode == 'run':
        if not check_changes_commited():
            sys.exit("Please commit all changes!")

    return train_csv, test_csv, emb_path, cache


def main(args, train_csv, test_csv, embedding, cache):
    train, test, text, qid = preprocess(train_csv, test_csv, args.tokenizer, embedding, cache)

    # split train dataset
    random.seed(args.seed)
    k = args.kfold
    if k:
        data_iter = train.split_kfold(k, is_test=True, random_state=random.getstate())
    else:
        data_iter = train.split(args.split_ratio, random_state=random.getstate())

    # iterate through folds
    for d in data_iter:
        print(len(d))
        if len(d) == 2:
            train, val = d
        else:
            train, val, test = d
        train_iter, val_iter, test_iter = iterate(train, val, test, args.batch_size)
        eval_every = int(len(list(iter(train_iter))) / args.n_eval)
        dataloaders = train_iter, val_iter, test_iter

        # choose model, optimizer, lr scheduler and loss function
        if args.model == 'BiLSTM':
            model = BiLSTM(text.vocab.vectors,
                           lstm_layer=args.n_layers,
                           padding_idx=text.vocab.stoi[text.pad_token],
                           hidden_dim=args.hidden_dim,
                           dropout=args.dropout).cuda()
        if args.optim == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optim == 'AdamW':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.99))
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
        loss_function = nn.BCEWithLogitsLoss()
        learn = Learner(model, dataloaders, loss_function, optimizer, scheduler, args)
        learn.fit(args.epoch, eval_every, args.f1_tresh, args.early_stop, args.warmup_epoch)

        # predict test labels
        learn.load()
        test_label, _, test_ids, tresh = learn.predict_labels(is_test=True, tresh=[0.01, 0.5, 0.01])
        if len(d) == 3:
            test_loss_old, test_f1_old = learn.evaluate(learn.test_dl, args.f1_tresh)
            print('Test results at point with best va lidation f1:', test_loss_old, test_f1_old)
        learn.record()
    test_ids = [qid.vocab.itos[i] for i in test_ids]
    submit(test_ids, test_label)


if __name__ == '__main__':
    args = parse_script_args()
    train_csv, test_csv, emb_path, cache = analyze_args(args)
    main(args, train_csv, test_csv, emb_path, cache)
