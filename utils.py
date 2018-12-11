import pandas as pd
import time
from datetime import timedelta

def submit(test_ids, prediciton, subm_name='submission.csv'):
    sub_df = pd.DataFrame()
    sub_df['qid'] = test_ids
    sub_df['prediction'] = prediciton
    sub_df.to_csv(subm_name, index=False)

    print('Predictions saved in submission.csv file')


def f1_metric(tp, n_targs, n_preds):
    if n_preds == 0 or n_targs == 0 or tp == 0:
        f1 = 0
    else:
        prec = tp/n_preds
        rec = tp/n_targs
        f1 = 2 * rec * prec / (rec + prec)
    return f1


def print_duration(time_start, message):
    time_end = time.time()
    seconds = int(time_end - time_start)
    tr_time = timedelta(seconds=seconds)
    print(f'{message}{tr_time}')