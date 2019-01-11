import numpy as np


def ens_mean(preds):
    return np.mean(np.array(preds), 0)


methods = {'mean': ens_mean}
