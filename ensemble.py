#!/usr/bin/env python
# Script reads predictions from csv files and ensembles predictions.

import argparse
import pandas as pd
import os
import csv
import numpy as np
from sklearn.metrics import f1_score

from ens_model_dict import model_dict
from learner import format_info, choose_thresh
from ens_methods import methods
from utils import dict_to_csv, submit, pred_to_csv


class Ensemble:
    ens_record_path = 'notes/ensemble.csv'

    def __init__(self, val_pred_paths, model_names, test_pred_paths=None, pred_dirs=None):
        self.val_pred_paths = val_pred_paths
        self.test_pred_paths = test_pred_paths
        self.pred_dirs = [os.path.dirname(p) for p in self.val_pred_paths] if not pred_dirs else pred_dirs
        self.model_names = model_names

    @classmethod
    def from_names(cls, model_names):
        model_args = 'names'
        val_pred_paths = [get_pred_path(m, 'val_probs.csv', model_args=model_args) for m in model_names]
        test_pred_paths = [get_pred_path(m, 'test_probs.csv', model_args=model_args) for m in model_names]
        return cls(val_pred_paths, model_names=model_names, test_pred_paths = test_pred_paths)

    @classmethod
    def from_dirs(cls, model_dirs):
        model_args = 'dirs'
        val_pred_paths = [get_pred_path(d, 'val_probs.csv', model_args=model_args) for d in model_dirs]
        test_pred_paths = [get_pred_path(d, 'test_probs.csv', model_args=model_args) for d in model_dirs]
        return cls(val_pred_paths, model_dirs,test_pred_paths, model_dirs)

    @classmethod
    def from_cv(cls, models, k=5, model_args='names'):
        """ For each single_model_dir looks for k-1 additional model dirs created right after model_dir.
        Then creates dir f'{single_model_dir}_cv' and creates in this dir 2 csv files val_probs_cv.csv" and test_probs_cv.csv.
        Writes concatenation of k val_probs.csv and k test_probs.csv into these  2 files."""
        single_model_dirs = [get_pred_dir(m, model_args) for m in models]
        head_dir = os.path.dirname(single_model_dirs[0])
        cv_head_dir = os.path.join(head_dir, 'cv')
        k_model_dirs = [find_k_dirs(d, k) for d in single_model_dirs]
        val_cv_paths, test_cv_paths = [], []
        for dirs in k_model_dirs:
            cv_dir = f'{os.path.basename(dirs[0])}_cv'
            cv_dir = os.path.join(cv_head_dir, cv_dir)
            os.makedirs(cv_dir, exist_ok=True)
            val_cv_path = os.path.join(cv_dir, 'val_probs_cv.csv')
            test_cv_path = os.path.join(cv_dir, 'test_probs_cv.csv')
            val_cv_paths.append(val_cv_path)
            test_cv_paths.append(test_cv_path)
            if not os.path.exists(val_cv_path):
                mode = 'w'
                for d in dirs:
                    val_data = load_pred_from_csv(os.path.join(d, 'val_probs.csv'))
                    pred_to_csv(*val_data, fpath=val_cv_path, mode=mode)
                    try:
                        test_data = load_pred_from_csv(os.path.join(d, 'test_probs.csv'))
                        pred_to_csv(*test_data, fpath=test_cv_path, mode=mode)
                    except:
                        'cant load test data'
                    if mode == 'w':
                        mode = 'a'
        return cls(val_cv_paths, models, test_cv_paths, single_model_dirs)

    def __call__(self, method, thresh, method_params=None):
        # find best method parameters based on validation data
        y_preds = []
        last_ids = None
        for pp in self.val_pred_paths:
            ids, y_prob, y_true = load_pred_from_csv(pp)
            y_preds.append(y_prob)
            if last_ids is None:
                last_ids = ids
            else:
                if not np.array_equal(last_ids, ids):
                    raise Exception('Prediction ids should be the same for ensemble')

        val_ens_prob = methods[method](y_preds, y_true, method_params)  # target probability after ensembling
        os.makedirs('./ensemble', exist_ok=True)
        try:
            pred_to_csv(ids, val_ens_prob, y_true, fpath=os.path.join('./ensemble', ' '.join(self.model_names)))
        except:
            print('cant save ensemble predictions for validation data')
        thresh, max_f1 = self.evaluate_ensemble(val_ens_prob, y_true, thresh)
        self.record(max_f1, thresh, method)
        # predict test labels and save submission
        # try:
        self.predict_test(method, thresh, method_params)
        # except:
        # print("can't predict test data")

    def predict_test(self, method, thresh, method_params):
        y_preds = []
        last_ids = None
        for pp in self.test_pred_paths:
            ids, y_prob, _ = load_pred_from_csv(pp)
            y_preds.append(y_prob)
            if last_ids:
                if np.array_equal(last_ids, ids):
                    raise Exception('Prediction ids should be the same for ensemble')
        test_ens_prob = methods[method](y_preds, args=method_params)  # target probability after ensembling
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
        model_infos = []  # partial model descriptions
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


def find_k_dirs(model_dir, k):
    head_dir = os.path.dirname(model_dir)
    all_entries = [os.path.join(head_dir, d) for d in os.listdir(head_dir)]
    all_dirs = [d for d in all_entries if os.path.isdir(d)]
    all_dirs.sort(key=lambda x: os.path.getctime(x))
    dir_idx = all_dirs.index(model_dir)
    k_dirs = all_dirs[dir_idx:dir_idx + k]
    return k_dirs


def get_pred_path(m, pred_file_name, model_args='names'):
    dir = get_pred_dir(m, model_args)
    path = os.path.join(dir, pred_file_name)
    return path


def get_pred_dir(m, model_args='names'):
    if model_args == 'names':
        dir = os.path.join('./models', model_dict[m][0])
    elif model_args == 'dirs':
        dir = m
    else:
        raise Exception('model_args should be names or dirs')
    return dir


def load_pred_from_csv(pred_path):
    df = pd.read_csv(pred_path)
    qid = df['qid']
    probs = df['prediction']
    if len(df.columns) == 3:
        true = df['true_label']
    else:
        true = pd.DataFrame([None] * len(df))
    column_values = [d.values for d in [qid, probs, true]]
    return column_values


def ens_parser(add_help=True):
    parser = argparse.ArgumentParser(add_help=add_help)
    arg = parser.add_argument
    arg('--models', '-m', nargs='+', type=str, default=['glove', 'wnews', 'paragram'])
    arg('--model_args', '-ma', type=str, default='names', choices=['names', 'dirs', 'paths'])
    arg('-k', default=None, type=int)
    arg('--method', '-mth', default='mean', type=str, choices=['mean', 'weight', 'stack'])
    arg('--weights', '-w', nargs='+', default=None, type=float)
    arg('--thresh', '-th', nargs='+', default=[0.1, 0.5, 0.01], type=float)
    return parser


def parse_ens_args():
    parser = ens_parser()
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_ens_args()
    if args.k:
        ens = Ensemble.from_cv(args.models, args.k, args.model_args)
    elif args.model_args == 'names':
        ens = Ensemble.from_names(args.models)
    elif args.model_args == 'dirs':
        ens = Ensemble.from_dirs(args.models)
    else:
        ens = Ensemble(args.models, model_names=args.models)
    ens(args.method, args.thresh, args)
