import argparse
import numpy as np


def evaluate_prob_array(ids, final_pred):
    pass


def get_model_path():
    pass


def load_pred_from_csv(m):
    model_path = get_model_path(m)
    pass


def ens(models, is_kfold, combine):
    preds = []
    last_ids = None
    for m in models:
        ids, p = load_pred_from_csv(m)
        preds.append(p)
        if last_ids:
            if last_ids != ids:
                raise Exception('Prediction ids should be the same for ensemble')
    if combine == 'mean':
        final_pred = np.mean(np.array(preds), 0)
    evaluate_prob_array(ids, final_pred)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--models', '-m', nargs='+', type=str)
    arg('-k', action='store_true')
    arg('--combine', '-c', default='mean', type=str, choices=['mean', 'weight', 'stack'])
    #arg('--seed', default=2018, type=int)
    args = parser.parse_args()

    ens(args.models, args.k, args.combine)