import sys
from ensemble import load_pred_from_csv
import pandas as pd
import os

class Analyze:
    def __init__(self, train_df, thresh):
        self.train_df = train_df
        self.thresh = thresh
        self.dir = './analyze'
        self.train_sorted = None
        self.incorrect = train_df[train_df['label'] != train_df['target']]
        self.correct = train_df[train_df['label'] == train_df['target']]
        self.subset = ['question_text', 'prediction']

    def random_incorrect(self, n, label):
        # choose incorrect indexes
        # random_choice of indexes
        # save n  text to file 'random_incorrect_label'
        incorrect_class = self.incorrect[self.incorrect['target'] == label]
        sample = incorrect_class.sample(n=n)
        sample = sample[self.subset]
        sample.to_csv(os.path.join(self.dir, f'random_incorrect_{label}.csv'))

    def most_incorrect(self, n, label):
        # choose incorrect indexes
        # sort y_prob with indexes
        # choose top n indexes
        # save n text to file 'most_incorrect_label'
        incorrect_class = self.incorrect[self.incorrect['target'] == label]
        incorrect_class_sorted = incorrect_class.sort_values(by='prediction')
        if label == 1:
            sample = incorrect_class_sorted[:n]
        else:
            sample = incorrect_class_sorted[-n:]
        sample = sample[self.subset]
        sample.to_csv(os.path.join(self.dir, f'most_incorrect_{label}.csv'))

    def most_doubt_incorrect(self, n, label):
        # choose incorrect indexes
        # sort probs, indexes by abs(thresh - prob)
        # choose top n indexes
        # save n text to file 'most_doubt_incorrect_label'
        pass

    def run_all(self, n):
        # run all functions for both labels
        pass

if __name__ == '__main__':
    pred_path = sys.argv[1]
    n = int(sys.argv[2])
    thresh = float(sys.argv[3])
    train_csv = './data/train.csv'
    pred_df = pd.read_csv(pred_path)
    pred_df['prediction'] = pred_df['prediction'].astype(float)
    y_prob = pred_df['prediction'].values
    y_label = (y_prob > thresh).astype(int)
    pred_df.drop(columns='true_label', inplace=True)
    pred_df['label'] = y_label
    train_df = pd.read_csv(train_csv)
    train_df['target'] = train_df['target'].astype(int)
    train_df = train_df.merge(pred_df)
    analyze = Analyze(train_df, thresh)
    analyze.random_incorrect(n, label=1)
    analyze.random_incorrect(n, label=0)
    # random_correct(n, label=1)
    analyze.most_incorrect(n, label=1)
    analyze.most_incorrect(n, label=0)
    # most_correct(n, label=1)
    # most_doubt_incorrect(n, label=1)
    # most_doubt_correct(n, label=1)