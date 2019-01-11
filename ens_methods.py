import numpy as np


def ens_mean(preds, args):
    return np.mean(np.array(preds), 0)


def ens_weight(preds, args):
    weights = args.weights
    preds = np.array(preds)
    if weights:
        weights = np.array(weights)
        preds = np.transpose(preds)
        final_pred = np.transpose(preds.dot(weights))
        return final_pred
    else:
        pass
    return np.mean(np.array(preds), 0)


methods = {'mean': ens_mean,
           'weight': ens_weight}
