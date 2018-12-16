import subprocess

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


def get_hash():
    hash = subprocess.check_output(['git', 'describe', '--always'])
    hash = hash.decode("utf-8")[1:-1]
    return hash


def str_date_time():
    struct_time = time.localtime()
    date_time = time.strftime('%b_%d_%Y__%H:%M:%S', struct_time)
    return date_time


def dict_to_csv(dict, csvname, mode, orient, reverse):
    if orient == 'index':
        df = pd.DataFrame.from_dict(dict, orient='index')
        df.to_csv(csvname, header=False, mode=mode)
    if orient == 'columns':
        df = pd.DataFrame(dict, index=[0])
        if reverse: #reverse dataframe columnes
            df = df.iloc[:, ::-1]
        df.to_csv(csvname, index=False, mode=mode)
    # TODO: append rows considering columns names


def check_changes_commited():
    message = subprocess.check_output(['git', 'status'])
    message_str = str(message.decode("utf-8"))
    required_message = 'On branch master\nnothing to commit, working tree clean\n'
    if message_str == required_message:
        status = True
    else:
        status = False
    return status

if __name__ == '__main__':
    git_tree_clean()