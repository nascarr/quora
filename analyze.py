import sys
from ensemble import load_pred_from_csv

if __name__ == '__main__':
    file_path = sys.argv[1]
    n = sys.argv[2]
    ids, y_label, y_true = load_pred_from_csv(file_path)
    random_incorrect(n, label=1)
    random_correct(n, label=1)
    most_incorrect(n, label=1)
    most_correct(n, label=1)
    most_doubt(n, label=1)