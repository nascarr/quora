#!/usr/bin/env python
# Script trains neural network models according to parsed arguments.

import argparse
import os
import random
import sys

import torch.nn as nn
import torch.optim as optim

from choose import choose_model, choose_optimizer
from create_test_datasets import reduce_embedding, reduce_datasets
from learner import Learner, choose_thresh
from preprocess import Data, iterate
from utils import submit, check_changes_commited, pred_to_csv


def parse_main_args(main_args=None):
    # parses arguments for main() function
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--machine', default='dt', choices=['dt', 'kaggle'], help='Local machine: dt. Kaggle kernel: kaggle.')
    arg('--mode', default='run', choices=['test', 'run'], help='Main mode: run. Test mode: test.')

    # data preprocessing params
    arg('--kfold', '-k', type=int, help='K-fold cross-validation.')
    arg('--split_ratio', '-sr', nargs='+', default=[0.8], type=float, help='Split ratio.')
    arg('--test', action='store_true', help='If present split data in train-val-test else split in train-val.')
    arg('--seed', default=2018, type=int, help='Seed for data split.')
    arg('--tokenizer', '-t', default='spacy', help='Tokenizer. See tokenizers.py.')
    arg('--embedding', '-em', default=['glove'], nargs='+', choices=['glove', 'gnews', 'paragram', 'wnews'], help='Embedding.')
    arg('--max_vectors', '-mv', default=5000000, type=int, help='Load no more than max_vectors number of embedding vectors.')
    arg('--no_cache', action='store_true', help='Don\'t cache embeddings.')
    arg('--var_length', '-vl', action = 'store_false', help='Variable sequence length in batches.') 
    arg('--unk_std', '-us', default = 0.001, type=float, help='Standart deviation for initialization\
                                                               of tokens without embedding vector.')
    arg('--stratified', '-s', action='store_true', help='Stratified split.')

    # training params
    arg('--optim', '-o', default='Adam', choices=['Adam', 'AdamW'], help='Optimizer. See choose.py')
    arg('--epoch', '-e', default=7, type=int, help='Number of epochs.')
    arg('--lr', '-lr', default=1e-3, type=float, help='Initial learning rate.')
    arg('--lrstep', default=[3], nargs='+', type=int, help='Steps when lr multiplied by 0.1.') # 
    arg('--batch_size', '-bs', default=512, type=int, help='Batch size.')
    arg('--n_eval', '-ne', default=1, type=int, help='Number of validation set evaluations during 1 epoch.')
    arg('--warmup_epoch', '-we', default=2, type=int, help='Number of epochs without embedding tuning.')
    arg('--early_stop', '-es', default=2, type=int, help='Stop training if no improvement during this number of epochs.')
    arg('--f1_tresh', '-ft', default=0.335, type=float, help='Threshold for calculation of F1-score.')
    arg('--clip', type=float, default=1, help='Gradient clipping.')

    # model params
    arg('--model', '-m', default='BiLSTMPool', help='Model name. See models.py.')
    arg('--n_layers', '-n', default=2, type=int, help='Number of RNN layers in model.')
    arg('--hidden_dim', '-hd', type=int, default=100, help='Hidden dimension for RNN.')
    arg('--dropout', '-d', type=float, default=0.2, help='Dropout probability.')

    if main_args:
        args = parser.parse_args(main_args)
    else:
        args = parser.parse_args()
    return args

def get_emb_path(emb_name):
    # returns path for embedding file according to embedding name: glove, paragram, gnews or wnews.
    if emb_name == 'glove':
        emb_path = 'glove.840B.300d/glove.840B.300d.txt'
    elif emb_name == 'paragram':
        emb_path = 'paragram_300_sl999/paragram_300_sl999.txt'
    elif emb_name == 'gnews':
        emb_path = 'GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
    elif emb_name == 'wnews':
        emb_path = 'wiki-news-300d-1M/wiki-news-300d-1M.vec'
    return emb_path

def preprocess_args(args):
    # preprocesses args

    # get embedding paths for multiple embeddings in the list args.embedding
    emb_paths = []
    for emb_name in args.embedding:
        emb_path = get_emb_path(emb_name)
        emb_paths.append(emb_path)
    # if machine is local computer
    if args.machine == 'dt':
        data_dir = cache = './data'
        if args.mode == 'run':
            if not check_changes_commited():
                sys.exit("Please commit all changes!")
    # if machine is cloud kernel at kaggle.com
    elif args.machine == 'kaggle':
        data_dir = '../input'
        cache = '.'
        args.no_cache = True

    train_csv = os.path.join(data_dir, 'train.csv')
    test_csv = os.path.join(data_dir, 'test.csv')
    emb_paths = [os.path.join(data_dir, 'embeddings', emb_path) for emb_path in emb_paths]

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
    return train_csv, test_csv, emb_paths, cache


def job(args, train_csv, test_csv, embeddings, cache):
    """ Reads data, makes preprocessing, trains model and records results.
        Gets args as argument and passes values of it's fields to functions."""

    data = Data(train_csv, test_csv, cache)

    # read and preprocess data
    to_cache = not args.no_cache
    data.read_embedding(embeddings, args.unk_std, args.max_vectors, to_cache)
    data.preprocess(args.tokenizer, args.var_length)
    data.embedding_lookup() if present split data in train-val-test else split train-val

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
        pred_to_csv(val_ids, y_pred, y_true)
        # choose best threshold for val predictions
        best_th, max_f1 = choose_thresh(y_pred, y_true, [0.1, 0.5, 0.01], message=True)
        learn.recorder.append_info({'best_th': best_th, 'max_f1': max_f1})


        # predict test labels
        test_label, test_prob,_, test_ids, tresh = learn.predict_labels(is_test=True, thresh=args.f1_tresh)
        if args.test:
            test_loss, test_f1, _, _, _ = learn.evaluate(learn.test_dl, args.f1_tresh)
            learn.recorder.append_info({'test_loss': test_loss, 'test_f1': test_f1}, message='Test set results: ')

        # save test predictions to submission.csv
        test_ids = [data.qid.vocab.itos[i] for i in test_ids]
        submit(test_ids, test_label, test_prob)
        record_path = learn.recorder.record(fold)  # directory path with all records
        print('\n')
    return record_path

def main(main_args=None):
    # Parse and preprocess args. Run job() with preprocessed args.
    args = parse_main_args(main_args)
    train_csv, test_csv, emb_paths, cache = preprocess_args(args)
    record_path = job(args, train_csv, test_csv, emb_paths, cache)
    return record_path


if __name__ == '__main__':
    # parse args and train model
    main()
