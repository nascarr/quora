import subprocess
import pandas as pd
import numpy as np
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import glob
import shutil


def _submit(test_ids, predictoins, subm_name):
    sub_df = pd.DataFrame()
    sub_df['qid'] = test_ids
    sub_df['prediction'] = predictoins
    sub_df.to_csv(subm_name, index=False)


def submit(test_ids, labels, probs, subm_name='submission.csv'):
    _submit(test_ids, labels, subm_name)
    _submit(test_ids, probs, 'test_probs.csv')
    print(f'predictions saved in {subm_name}, test_probs.csv file')


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
    minutes = seconds/60
    return minutes


def get_hash():
    hash = subprocess.check_output(['git', 'describe', '--always'])
    hash = hash.decode("utf-8")[1:-1]
    return hash


def str_date_time():
    struct_time = time.localtime()
    date_time = time.strftime('%b_%d_%Y__%H:%M:%S', struct_time)
    return date_time


def dict_to_csv(dict, csvname, mode, orient, reverse=False, header=True):
    if orient == 'index':
        df = pd.DataFrame.from_dict(dict, orient='index')
        df.to_csv(csvname, header=False, mode=mode)
    if orient == 'columns':
        df = pd.DataFrame(dict, index=[0])
        if reverse: #reverse dataframe columnes
            df = df.iloc[:, ::-1]
        df.to_csv(csvname, index=False, mode=mode, header=header)
    # TODO: append rows considering columns names


def check_changes_commited():
    message = subprocess.check_output(['git', 'status'])
    message_last_line = str(message.decode("utf-8")).split('\n')[-2]
    required_last_line = 'nothing to commit, working tree clean'
    if message_last_line == required_last_line:
        status = True
    else:
        status = False
    return status


def save_plot(record, key, n_eval, fname):
    y = [o[key] for o in record]
    x = np.array(range(len(y))) / n_eval
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    ax.set_xlabel('epoch')
    ax.set_ylabel(key)
    fig.savefig(fname)


def save_plots(records, keys, labels, n_eval):
    for k in keys:
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel('epoch')
        ax.set_ylabel(k)
        for r, l in zip(records, labels):
            y = [o[k] for o in r]
            x = np.array(range(len(y))) / n_eval
            ax.plot(x, y, label=l)
        plt.legend()
        fname = k + '.png'
        fig.savefig(fname)
        plt.close()


def copy_files(arg_list, dest_dir):
    for a in arg_list:
        for file in glob.glob(a):
            shutil.copy(file, dest_dir)
