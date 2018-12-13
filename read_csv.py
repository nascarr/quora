# This file is to measure data reading time
import csv
import os
from utils import print_duration
import time

def read_csv(file):
    with open(file) as f:
        row_list = []
        csvreader = csv.reader(f, delimiter=',')
        for row in csvreader:
            row_list.append(row[1])
    return row_list

data_dir = './data'
train_csv = os.path.join(data_dir, 'train.csv')
test_csv = os.path.join(data_dir, 'test.csv')
time_start = time.time()
tr_rows = read_csv(train_csv)
test_rows = read_csv(test_csv)
print_duration(time_start, 'Time to read data')
print(tr_rows[1000000], test_rows[50000])