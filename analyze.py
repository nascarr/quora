# Analyze class is designed for analysis of model predictions. It has methods for writing into csv file
# n random correct predictions, n random incorrect predictions, n most correct predictions
# (sorted by predicted probability), n most incorrect predictions and n 'most doubt' predictions with predicted
# probability closest to threshold.

import sys
import pandas as pd
import os

class Analyze:
    def __init__(self, train_df, thresh):
        self.train_df = train_df
        self.thresh = thresh
        self.dir = './analyze'
        self.train_sorted = None
        self.subset = ['question_text', 'prediction']
        self.incorrect = train_df[train_df['label'] != train_df['target']]
        self.correct = train_df[train_df['label'] == train_df['target']]

    def csv_path(self, name, n, label):
        path = os.path.join(self.dir, f'{name}_{label}_{n}.csv')
        return path

    def random_sample(self, df, n):
        sample = df.sample(n=n)
        return sample

    def most_sample(self, df, n, label):
        df_sorted = df.sort_values(by='prediction')
        if label == 1:
            sample = df_sorted[:n]
        else:
            sample = df_sorted[-n:]
        return sample

    def doubt_sample(self, df, n):
        df = df.assign(metric=lambda x: abs(self.thresh - x['prediction']))
        df_sorted = df.sort_values(by='metric')
        sample = df_sorted[:n].drop('metric', axis=1)
        return sample

    def random_incorrect(self, n, label):
        # choose incorrect predictions
        # random_choice of predictions
        # save n text entries to file 'random_incorrect_label'
        incorrect_class = self.incorrect[self.incorrect['target'] == label][self.subset]
        sample = self.random_sample(incorrect_class, n)
        sample.to_csv(self.csv_path('random_incorrect', n, label))

    def most_incorrect(self, n, label):
        # choose incorrect predictions
        # sort y_prob by prediction probability
        # choose top or bottom n indexes
        # save n text entries to file 'most_incorrect_label'
        incorrect_class = self.incorrect[self.incorrect['target'] == label][self.subset]
        sample = self.most_sample(incorrect_class, n, label)
        sample.to_csv(self.csv_path('most_incorrect', n, label))

    def most_doubt_incorrect(self, n, label):
        # choose incorrect predictions
        # sort probs, indexes by abs(thresh - prob)
        # choose top or bottom n indexes
        # save n text entries to file 'most_doubt_incorrect_label'
        correct_class = self.correct[self.correct['target'] == label][self.subset]
        sample = self.doubt_sample(correct_class, n)
        sample.to_csv(self.csv_path('doubt_correct', n, label))

    def random_correct(self, n, label):
        # choose incorrect predictions
        # random_choice of predictions
        # save n text entries to file 'random_incorrect_label'
        correct_class = self.correct[self.correct['target'] == label][self.subset]
        sample = self.random_sample(correct_class, n)
        sample.to_csv(self.csv_path('random_correct', n, label))

    def most_correct(self, n, label):
        # choose incorrect predictions
        # sort y_prob by prediction probability
        # choose top or bottom n indexes
        # save n text entries to file 'most_incorrect_label'
        correct_class = self.correct[self.correct['target'] == label][self.subset]
        sample = self.most_sample(correct_class, n, label)
        sample.to_csv(self.csv_path('most_correct', n, label))

    def most_doubt_correct(self, n, label):
        # choose incorrect predictions
        # sort probs, indexes by abs(thresh - prob)
        # choose top or bottom n indexes
        # save n text entries to file 'most_doubt_incorrect_label'
        correct_class = self.correct[self.correct['target'] == label][self.subset]
        sample = self.doubt_sample(correct_class, n)
        sample.to_csv(self.csv_path('doubt_correct', n, label))

    def run_all(self, n):
        # run all functions for both labels
        for l in (0, 1):
            self.random_correct(n, l)
            self.most_doubt_correct(n, l)
            self.most_correct(n, l)
            self.random_incorrect(n, l)
            self.most_doubt_incorrect(n, l)
            self.most_incorrect(n, l)
            

if __name__ == '__main__':
    # parse script arguments: prediction path, number of text entries to save, threshold
    pred_path = sys.argv[1]
    n = int(sys.argv[2])
    thresh = float(sys.argv[3])

    # build dataframe train_df
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

    # run all Analyze class methods
    analyze = Analyze(train_df, thresh)
    analyze.run_all(n)
    # most_correct(n, label=1)
    # most_doubt_incorrect(n, label=1)
    # most_doubt_correct(n, label=1)