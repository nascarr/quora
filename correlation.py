#!/usr/bin/env python

import pandas as pd
import sys
from ensemble import get_pred_path

first_file = sys.argv[1]
second_file = sys.argv[2]

def corr(first_model, second_model):
  pred_file_name = 'val_probs.csv'
  first_path, second_path = [get_pred_path(m, pred_file_name) for m in [first_model, second_model]]
  first_df = pd.read_csv(first_path,index_col=0)
  second_df = pd.read_csv(second_path,index_col=0)
  # assuming first column is `prediction_id` and second column is `prediction`
  prediction = first_df.columns[0]
  # correlation
  print("Finding correlation between: {} and {}".format(first_file,second_file))
  print("Column to be measured: {}".format(prediction))
  print("Pearson's correlation score: {}".format(first_df[prediction].corr(second_df[prediction],method='pearson')))
  print("Kendall's correlation score: {}".format(first_df[prediction].corr(second_df[prediction],method='kendall')))
  print("Spearman's correlation score: {}".format(first_df[prediction].corr(second_df[prediction],method='spearman')))

if __name__ == '__main__':
  corr(first_file, second_file)
