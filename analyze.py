import sys
from ensemble import load_pred_from_csv
import pandas as pd

class Analyze:
    def __init__(self):
        self.ids
        self.text
        self.y_label
        self.y_true
        self.y_prob
        self.thresh
        self.y_prob_sorted
        self.incorrect # indexes of incorrect labels
        self.correct # indexes of correct labels

    def random_incorrect(self, n, label):
        # choose incorrect indexes
        # random_choice of indexes
        # save n  text to file 'random_incorrect_label'

    def most_incorrect(self, n, label):
        # choose incorrect indexes
        # sort y_prob with indexes
        # choose top n indexes
        # save n text to file 'most_incorrect_label'
        pass

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
    n = sys.argv[2]
    thresh = sys.argv[3]
    train_csv = './data/train.csv'
    pred_df = pd.read_csv(pred_path)
    y_prob = pred_df['prediction'].values.tolist()
    y_label = (y_prob > thresh).astype(int)
    pred_df.drop(columns='true_label', inplace=True)
    pred_df['label'] = y_label
    pred_df.set_index('qid', inplace=True)
    train_df = pd.read_csv(train_csv).set_index('qid')

    analyze = Analyze(train_df, thresh)
    analyze.random_incorrect(n, label=1)
    #random_correct(n, label=1)
    #most_incorrect(n, label=1)
    #most_correct(n, label=1)
    #most_doubt_incorrect(n, label=1)
    #most_doubt_correct(n, label=1)