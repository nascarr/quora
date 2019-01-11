#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
from ens_model_list import model_dict
import os
import csv
from learner import format_info
from ens_methods import methods
from sklearn.metrics import f1_score
from utils import dict_to_csv
import warnings
class Ensemble:

    ens_record_path = 'notes/ensemble.csv'
    record_path = 'notes/records.csv'

    def __init__(self, models, is_kfold=False):
        self.models_ = [model_dict[m][0] for m in models]
        self.descriptions = [model_dict[m][1] for m in models]
        self.methods = methods

    def __call__(self, method='mean', tresh=[0.1, 0.5, 0.01]):
        preds = []
        last_ids = None
        for m in self.models:
            ids, p, true = load_pred_from_csv(m)
            preds.append(p)
            if last_ids:
                if last_ids != ids:
                    raise Exception('Prediction ids should be the same for ensemble')
        final_pred = methods[method](preds)
        tresh, max_f1 = self.evaluate_ensemble(final_pred, true, tresh)
        self.record(max_f1, tr, method)

    @staticmethod
    def evaluate_ensemble(final_pred, true, thresh):
        if type(thresh) == float:
            f1 = f1_score(true, final_pred > thresh)
            return thresh, f1
        elif type(thresh) == list:
            optim_thr, max_f1 = choose_tresh(final_pred, true, thresh)
            print('Best threshold for ensemble prediction is {:.4f} with F1 score: {:.4f}'.format(thr, max_f1))
            return thr, max_f1
        else:
            raise Exception('Threshold must be float or list of 3 floats')

    def record(self, max_f1, tresh, method):
        info = format_info({'max_f1': max_f1, 'tresh': tresh})
        info = {'method': method, **info}
        model_descrps = []
        # copy partial models descriptions
        with open(self.record_path, 'r') as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if idx in self.descriptions:
                    model_descrps.append(row)
        with open(self.ens_record_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerows(model_descrps)
        dict_to_csv(info, self.ens_record_path, 'a', 'columns', reverse=False, header=True)


def val_pred_to_csv(ids, y_pred, y_true, fname='val_probs.csv'):
    df = pd.DataFrame()
    df['qid'] = ids
    df['prediction'] = y_pred
    df['true_label'] = y_true
    df.to_csv(fname, index=False)


def get_model_path(m):
    path = os.path.join('./models', model_dict[m])


def load_pred_from_csv(m):
    model_path = get_model_path(m)
    df = pd.DataFrame.from_csv(model_path)
    qid = df['qid']
    probs = df['prediction']
    true = df['true']
    qid, probs, true = [d.values for d in [qid, probs, true]]
    return qid, probs, true


def choose_tresh(probs, true, thresh_range):
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

    return th, tmp[2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--models', '-m', nargs='+', type=str)
    arg('-k', action='store_true')
    arg('--method', '-mth', default='mean', type=str, choices=['mean', 'weight', 'stack'])
    args = parser.parse_args()
    ens = Ensemble(arg.models, arg.k)
    ens(method=args.method)

