#!/usr/bin/env python

import argparse
import os
import random
import sys

import torch.nn as nn
import torch.optim as optim

from choose import choose_tokenizer, choose_model, choose_optimizer
from create_test_datasets import reduce_embedding, reduce_datasets
from ensemble import val_pred_to_csv
from learner import Learner, choose_thresh
from preprocess import Data, iterate
from utils import submit, check_changes_commited


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
    arg('--stratified', '-s', action='store_true')

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
    data_iter = data.split(args.kfold, args.split_ratio, args.stratified, args.test, args.seed)

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


if __name__ == '__main__':
    main()
