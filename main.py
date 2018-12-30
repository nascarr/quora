#!/usr/bin/env python

import os
import sys
import random
import torch.nn as nn
import torch.optim as optim
import argparse

from models import *
from preprocess import preprocess, iterate
from learner import Learner
from utils import submit, check_changes_commited
from create_test_datasets import reduce_embedding, reduce_datasets


def parse_script_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--machine', default='dt', choices=['dt', 'kaggle'])
    arg('--mode', default='run', choices=['test', 'run'])

    # data preprocessing params
    arg('--kfold', '-k', type=int)
    arg('--split_ratio', '-sr', nargs='+', default=[0.8], type=float)
    arg('--test', action='store_true') # if present split data in train-val-test else split train-val
    arg('--seed', default=2018, type=int)
    arg('--tokenizer', '-t', default='spacy', choices=['spacy'])
    arg('--embedding', '-em', default='glove', choices=['glove', 'google_news', 'paragram', 'wiki_news'])

    # training params
    arg('--optim', '-o', default='Adam', choices=['Adam', 'AdamW'])
    arg('--epoch', '-e', default=7, type=int)
    arg('--lr', '-lr', default=1e-3, type=float)
    arg('--lrstep', default=[3], nargs='+', type=int) # steps when lr multiplied by 0.1
    arg('--batch_size', '-bs', default=512, type=int)
    arg('--n_eval', '-ne', default=1, type=int, help='Number of validation set evaluations during 1 epoch')
    arg('--warmup_epoch', '-we', default=2, type=int, help='Number of epochs without fine tuning')
    arg('--early_stop', '-es', default=1, type=int, help='Stop training if no improvement during this number of epochs')
    arg('--f1_tresh', '-ft', default=0.335, type=float)
    arg('--clip', type=float, default=1, help='gradient clipping')

    # model params
    arg('--model', '-m', default='BiLSTM', choices=['BiLSTM', 'BiGRU', 'BiLSTMPool', 'BiLSTM_2FC', 'BiGRUPool', 'BiGRUPool_2FC', 'BiLSTMPool_2FC'])
    arg('--n_layers', '-n', default=2, type=int, help='Number of layers in model')
    arg('--hidden_dim', '-hd', type=int, default=100)
    arg('--dropout', '-d', type=float, default=0.2)

    args = parser.parse_args()
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

    # split ratio should be float if len == 1
    sr = args.split_ratio
    if len(sr) == 1:
        args.split_ratio = sr[0]
    return train_csv, test_csv, emb_path, cache


def main(args, train_csv, test_csv, embedding, cache):
    train, test, text, qid = preprocess(train_csv, test_csv, args.tokenizer, embedding, cache)

    # split train dataset
    random.seed(args.seed)
    k = args.kfold
    if k:
        data_iter = train.split_kfold(k, is_test=args.test, random_state=random.getstate())
    else:
        data_iter = train.split(args.split_ratio, random_state=random.getstate())

    # iterate through folds
    for fold, d in enumerate(data_iter):
        print(f'========== Fold {fold} ==========')
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
        if args.model == 'BiGRU':
            model = BiGRU(text.vocab.vectors,
                           lstm_layer=args.n_layers,
                           padding_idx=text.vocab.stoi[text.pad_token],
                           hidden_dim=args.hidden_dim,
                           dropout=args.dropout).cuda()
        if args.model == 'BiLSTMPool':
            model = BiLSTMPool(text.vocab.vectors,
                           lstm_layer=args.n_layers,
                           padding_idx=text.vocab.stoi[text.pad_token],
                           hidden_dim=args.hidden_dim,
                           dropout=args.dropout).cuda()
        if args.model == 'BiGRUPool':
            model = BiGRUPool(text.vocab.vectors,
                               lstm_layer=args.n_layers,
                               padding_idx=text.vocab.stoi[text.pad_token],
                               hidden_dim=args.hidden_dim,
                               dropout=args.dropout).cuda()
        if args.model == 'BiLSTM_2FC':
            model = BiLSTM_2FC(text.vocab.vectors,
                               lstm_layer=args.n_layers,
                               padding_idx=text.vocab.stoi[text.pad_token],
                               hidden_dim=args.hidden_dim,
                               dropout=args.dropout).cuda()

        if args.model == 'BiGRUPool_2FC':
            model = BiGRUPool_2FC(text.vocab.vectors,
                               lstm_layer=args.n_layers,
                               padding_idx=text.vocab.stoi[text.pad_token],
                               hidden_dim=args.hidden_dim,
                               dropout=args.dropout).cuda()

        if args.model == 'BiLSTMPool_2FC':
            model = BiLSTMPool_2FC(text.vocab.vectors,
                               lstm_layer=args.n_layers,
                               padding_idx=text.vocab.stoi[text.pad_token],
                               hidden_dim=args.hidden_dim,
                               dropout=args.dropout).cuda()

        if args.optim == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optim == 'AdamW':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.99))
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lrstep, gamma=0.1)
        loss_function = nn.BCEWithLogitsLoss()
        learn = Learner(model, dataloaders, loss_function, optimizer, scheduler, args)
        learn.fit(args.epoch, eval_every, args.f1_tresh, args.early_stop, args.warmup_epoch, args.clip)

        # predict test labels
        learn.recorder.load()
        test_label, _, test_ids, tresh = learn.predict_labels(is_test=True, tresh = [0.01, 0.5, 0.01])
        if args.test:
            test_loss, test_f1 = learn.evaluate(learn.test_dl, args.f1_tresh)
            print('Test loss and f1:', test_loss, test_f1)
        learn.recorder.record(fold)
    test_ids = [qid.vocab.itos[i] for i in test_ids]
    submit(test_ids, test_label)


if __name__ == '__main__':
    args = parse_script_args()
    train_csv, test_csv, emb_path, cache = analyze_args(args)
    main(args, train_csv, test_csv, emb_path, cache)
