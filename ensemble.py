#!/usr/bin/env python

import argparse
import pandas as pd
from ens_model_list import model_dict
import os
import csv
from learner import format_info, choose_thresh
from ens_methods import methods
from sklearn.metrics import f1_score
from utils import dict_to_csv


class Ensemble:

    ens_record_path = 'notes/ensemble.csv'

    def __init__(self, models, is_kfold=False):
        self.models = models
        self.descr_rows = self.descr_rows_for_models(models) # office index starts with 1
        self.methods = methods
        if models[0][-4:] == 'test':
            self.record_path = 'notes/test_records.csv'
        else:
            self.record_path = 'notes/records.csv'

    def __call__(self, args):
        thresh = args.thresh
        method = args.method
        preds = []
        last_ids = None
        for m in self.models:
            ids, p, true = load_pred_from_csv(m)
            preds.append(p)
            if last_ids:
                if last_ids != ids:
                    raise Exception('Prediction ids should be the same for ensemble')
        final_pred = methods[method](preds, args)
        thresh, max_f1 = self.evaluate_ensemble(final_pred, true, thresh)
        self.record(max_f1, thresh, method)

    @staticmethod
    def descr_rows_for_models(models):
        value_rows = [model_dict[m][1] - 1 for m in models]
        header_rows = [r - 1 for r in value_rows]
        return value_rows + header_rows

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

    def record(self, max_f1, tresh, method):
        info = format_info({'max_f1': max_f1, 'tresh': tresh})
        info = {'method': method, **info}
        model_descrps = ['']
        # copy partial models descriptions
        with open(self.record_path, 'r') as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if idx in self.descr_rows:
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


def get_pred_path(m):
    path = os.path.join('./models', model_dict[m][0], 'val_probs.csv')
    return path


def load_pred_from_csv(m):
    pred_path = get_pred_path(m)
    df = pd.read_csv(pred_path)
    qid = df['qid']
    probs = df['prediction']
    true = df['true_label']
    qid, probs, true = [d.values for d in [qid, probs, true]]
    return qid, probs, true


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--models', '-m', nargs='+', type=str, default=['glove', 'wnews', 'paragram', 'gnews'])
    arg('-k', action='store_true')
    arg('--method', '-mth', default='mean', type=str, choices=['mean', 'weight', 'stack'])
    arg('--weights', '-w', nargs='+', default=[0.9, 0.1], type=float)
    arg('--thresh', '-th', nargs='+', default=[0.1, 0.5, 0.01], type=float)
    args = parser.parse_args()
    ens = Ensemble(args.models, args.k)
    ens(args)

