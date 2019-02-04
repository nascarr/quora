import numpy as np
from sklearn.linear_model import LinearRegression


def ens_mean(preds, y_true=None, args=None):
    return np.mean(np.array(preds), 0)


def ens_weight(y_preds, y_true=None, args=None):
    weights = args.weights
    y_preds = np.transpose(np.array(y_preds))
    if not weights:
        X = y_preds
        y = np.transpose(y_true)
        reg = LinearRegression(fit_intercept=False).fit(X, y)
        print('lin reg score: ', reg.score(X, y))
        weights = reg.coef_
        print('weights: ', weights)
    weights = np.array(weights)
    final_pred = np.transpose(y_preds.dot(weights))
    return final_pred


methods = {'mean': ens_mean,
           'weight': ens_weight}
