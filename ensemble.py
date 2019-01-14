#!/usr/bin/env python

import argparse
import pandas as pd
from ens_model_list import model_dict
import os
import csv
from learner import format_info, choose_thresh
from ens_methods import methods
from sklearn.metrics import f1_score
from utils import dict_to_csv, submit


class Ensemble:

    ens_record_path = 'notes/ensemble.csv'

    def __init__(self, models, is_kfold=False, model_args='names'):
        self.pred_paths = [get_pred_path(m, model_args) for m in models]
        self.pred_dirs = [get_pred_dir(m, model_args) for m in models]

    def __call__(self, args):
        thresh = args.thresh
        method = args.method

        # find best method parameters based on validation data
        y_preds = []
        last_ids = None
        for pp in self.pred_paths:
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
        for pp in self.pred_paths:
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


def get_pred_path(m, model_args='names'):
    if model_args == 'names':
        path = os.path.join('./models', model_dict[m][0], 'val_probs.csv')
    elif model_args == 'paths':
        path = os.path.join(m, 'val_probs.csv')
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

if __name__ == '__main__':
    args = parse_ens_args()
    ens = Ensemble(args.models, args.k)
    ens(args)

