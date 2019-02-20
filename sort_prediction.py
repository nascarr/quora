#!/usr/bin/env python
# Sorts prediction at first file according to ids order at second file.

import sys
import os
from ensemble import load_pred_from_csv
from utils import pred_to_csv

first_file = sys.argv[1] # file to sort
second_file = sys.argv[2] # file as example of ids order

if __name__ == '__main__':
    ids, probs, true = load_pred_from_csv(first_file)
    first_dir = os.path.dirname(first_file)
    example_ids, _, _ = load_pred_from_csv(second_file)
    stoi = {e:i for i, e in enumerate(example_ids)}
    idxs = list(range(len(ids)))
    idxs.sort(key=lambda x: stoi[ids[x]])
    sorted_probs = probs[idxs]
    sorted_true = true[idxs]
    new_path = os.path.join(first_dir, 'val_probs_sorted.csv')
    pred_to_csv(example_ids, sorted_probs, sorted_true, fpath=new_path)

