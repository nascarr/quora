#!/usr/bin/env python

import os
import sys
import random
import torch.nn as nn
import torch.optim as optim
import argparse

from preprocess import preprocess, split, iterate
from tokenizers import WhitespaceTokenizer
from choose import choose_tokenizer, choose_model, choose_optimizer
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
    arg('--tokenizer', '-t', default='spacy', choices=['spacy', 'whitespace', 'custom'])
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
        if args.mode == 'run':
            if not check_changes_commited():
                sys.exit("Please commit all changes!")
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
    if args.mode == 'run':
        pass

    # split ratio should be float if len == 1
    sr = args.split_ratio
    if len(sr) == 1:
        args.split_ratio = sr[0]
    if len(sr) == 3:
        args.test = True
    return train_csv, test_csv, emb_path, cache


def main(args, train_csv, test_csv, embedding, cache):
    tokenizer = choose_tokenizer(args.tokenizer)
    train, test, text, qid = preprocess(train_csv, test_csv, tokenizer, embedding, cache)
    # split train dataset
    random.seed(args.seed)
    data_iter = split(train, args)

    # iterate through folds
    for fold, d in enumerate(data_iter):
        print(f'========== Fold {fold} ==========')
        # get dataloaders
        if len(d) == 2:
            train, val = d
        else:
            train, val, test = d
        train_iter, val_iter, test_iter = iterate(train, val, test, args.batch_size)
        dataloaders = train_iter, val_iter, test_iter

        # choose model, optimizer, lr scheduler and loss function
        model = choose_model(text, args)
        optimizer = choose_optimizer(filter(lambda p: p.requires_grad, model.parameters()), args)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lrstep, gamma=0.1)
        loss_function = nn.BCEWithLogitsLoss()
        learn = Learner(model, dataloaders, loss_function, optimizer, scheduler, args)
        eval_every = int(len(list(iter(train_iter))) / args.n_eval)
        learn.fit(args.epoch, eval_every, args.f1_tresh, args.early_stop, args.warmup_epoch, args.clip)

        # predict test labels
        learn.model, info = learn.recorder.load()
        test_label, _, test_ids, tresh = learn.predict_labels(is_test=True, tresh = args.f1_tresh)
        if args.test:
            test_loss, test_f1 = learn.evaluate(learn.test_dl, args.f1_tresh)
            learn.recorder.append_info({'test_loss': test_loss, 'test_f1': test_f1}, message='Test set results: ')
        learn.recorder.record(fold)
    test_ids = [qid.vocab.itos[i] for i in test_ids]
    submit(test_ids, test_label)


if __name__ == '__main__':
    args = parse_script_args()
    train_csv, test_csv, emb_path, cache = analyze_args(args)
    main(args, train_csv, test_csv, emb_path, cache)
