import argparse
import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext.data as data
import torchtext.vocab as vocab

from sklearn.metrics import f1_score
import warnings
import time
from datetime import timedelta



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




def preprocess(train_csv, test_csv, tokenizer, embeddings, cache):
    # types of csv columns
    time_start = time.time()
    text = data.Field(batch_first=True, tokenize=tokenizer)
    qid = data.Field()
    target = data.Field(sequential=False, use_vocab=False, is_target=True)

    # read and tokenize data
    print('Reading data.')
    train = data.TabularDataset(path=train_csv, format='csv',
                                fields={'qid': ('qid', qid),
                                        'question_text': ('text', text),
                                        'target': ('target', target)})
    test = data.TabularDataset(path=test_csv, format='csv',
                               fields={'qid': ('qid', qid),
                                       'question_text': ('text', text)})
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
                                    sort=True)
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
        y = self.hidden2label(self.dropout(torch.cat([c_n[i, :, :] for i in range(c_n.shape[0])], dim=1)))
        return y

class Learner:
    def __init__(self, model, dataloaders, loss_func, optimizer):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        if len(dataloaders) == 3:
            self.train_dl, self.val_dl, self.test_dl = dataloaders
        elif len(dataloaders) == 2:
            self.train_dl, self.val_dl = dataloaders
        elif len(dataloaders) == 1:
            self.train_dl = dataloaders
            self.val_dl = None

    def fit(self, epoch, eval_every, tresh, early_stop=1, warmup_epoch=2):
        print('Start training!')
        time_start = time.time()
        step = 0
        max_f1 = 0
        no_improve_epoch = 0
        no_improve_in_previous_epoch = False
        fine_tuning = False
        train_record = []
        val_record = []
        losses = []

        torch.backends.cudnn.benchmark = True
        for e in range(epoch):
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
                self.model.train()
                x = train_batch.text.cuda()
                y = train_batch.target.type(torch.Tensor).cuda()
                self.model.zero_grad()
                pred = self.model.forward(x).view(-1)
                loss = self.loss_func(pred, y)
                losses.append(loss.cpu().data.numpy())
                train_record.append(loss.cpu().data.numpy())
                loss.backward()
                self.optimizer.step()
                if step % eval_every == 0:
                    self.model.eval()
                    self.model.zero_grad()
                    val_loss = []
                    tp = 0
                    n_targs = 0
                    n_preds = 0

                    for val_batch in iter(self.val_dl):
                        val_x = val_batch.text.cuda()
                        val_y = val_batch.target.type(torch.Tensor).cuda()
                        val_pred = self.model.forward(val_x).view(-1)
                        val_loss.append(self.loss_func(val_pred, val_y).cpu().data.numpy())
                        val_label = (torch.sigmoid(val_pred).cpu().data.numpy() > tresh).astype(int)
                        val_y = val_y.cpu().data.numpy()
                        tp += sum(val_y + val_label == 2)
                        n_targs += sum(val_y)
                        n_preds += sum(val_label)
                    f1 = f1_metric(tp, n_targs, n_preds)

                    val_record.append({'step': step, 'loss': np.mean(val_loss), 'f1_score': f1})
                    print('epoch {:02} - step {:06} - train_loss {:.4f} - val_loss {:.4f} - f1 {:.4f}'.format(
                        e, step, np.mean(losses), val_record[-1]['loss'], f1))
                    if val_record[-1]['f1_score'] >= max_f1:
                        self.save(info={'step': step, 'epoch': e, 'train_loss': np.mean(losses),
                                        'val_loss': val_record[-1]['loss'], 'f1_score': val_record[-1]['f1_score']})
                        max_f1 = val_record[-1]['f1_score']
                        no_improve_in_previous_epoch = False

        m_info = self.load()
        print(f'Best model: {m_info}')
        print_duration(time_start, 'Training time: ')


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
            print('best threshold is {:.4f} with F1 score: {:.4f}'.format(tr, tmp[2]))
            return tr

        y_pred, y_true, ids = self.predict_probs(is_test=is_test)

        if type(tresh) == list:
            tresh = _choose_tr(self, *tresh)

        y_label = (np.array(y_pred) >= tresh).astype(int)
        return y_label, y_true, ids



    def save(self, info):
        os.makedirs('./models', exist_ok=True)
        torch.save(info, './models/best_model.info')
        torch.save(self.model, './models/best_model.m')

    def load(self):
        self.model = torch.load('./models/best_model.m')
        info = torch.load('./models/best_model.info')
        return info


def main(args, train_csv, test_csv, embedding, cache):
    train, test, text, qid = preprocess(train_csv, test_csv, args.tokenizer, embedding, cache)
    random.seed(args.seed)
    train, val = train.split(split_ratio=args.split_ratio, random_state=random.getstate())
    train_iter, val_iter, test_iter = iterate(train, val, test, args.batch_size)

    eval_every = len(list(iter(train_iter)))/args.n_eval
    model = BiLSTM(text.vocab.vectors, lstm_layer=2, padding_idx=text.vocab.stoi[text.pad_token], hidden_dim=128).cuda()
    # loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_w]).cuda())
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    dataloaders = train_iter, val_iter, test_iter
    learn = Learner(model, dataloaders, loss_function, optimizer)
    learn.fit(args.epoch, eval_every, args.f1_tresh, args.early_stop, args.warmup_epoch)

    # predict test labels
    learn.load()
    test_label, _, test_ids = learn.predict_labels(is_test=True, tresh=[0.01, 0.5, 0.01])
    test_ids = [qid.vocab.itos[i] for i in test_ids]
    submit(test_ids, test_label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--machine', default='dt', choices=['dt', 'kaggle'])
    arg('--mode', default='test', choices=['test', 'run'])
    arg('--epoch', '-e', default=5, type=int)
    arg('--lr', '-l', default=1e-3, type=float)
    arg('--batch_size', '-bs', default=512, type=int)
    arg('--n_eval', '-ne', default=1, type=int, help='Number of validation set evaluations during 1 epoch')
    arg('--warmup_epoch', '-we', default=2, type=int, help='Number of epochs without fine tuning')
    arg('--early_stop', '-es', default=1, type=int, help='Stop training if no improvement during this number of epochs')
    arg('--split_ratio', '-sr', default=0.8, type=float)
    arg('--seed', default=2018, type=int)
    arg('--tokenizer', '-t', default='spacy', choices=['spacy'])
    arg('--embedding', '-em', default='glove', choices=['glove', 'google_news','paragram', 'wiki_news'])
    arg('--f1_tresh', '-ft', default=0.33, type=float)

    args = parser.parse_args(args=[])


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
        args.batch_size = n_cut / 5

        if args.machine == 'kaggle':
            data_dir = '.'

        train_small_csv, test_small_csv = reduce_datasets([train_csv, test_csv], data_dir, n_cut)
        emb_small_path = reduce_embedding(emb_path, data_dir, n_cut_emb)
        train_csv, test_csv, emb_path = train_small_csv, test_small_csv, emb_small_path
    elif args.mode == 'run':
        pass  # all parameters for 'run' mode defined above

    main(args, train_csv, test_csv, emb_path, cache)
