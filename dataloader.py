from torchtext.data.dataset import *
import numpy as np


class MyTabularDataset(TabularDataset):
    """Subclass of torch.data.dataset.TabularDataset for k-fold cross-validation"""
    def split(self, split_ratio=0.8, stratified=False, strata_field='label',
              random_state=None):
        splits = super().split(split_ratio, stratified, strata_field, random_state)
        return [splits]

    def split_kfold(self, k, stratified=False, test=False, random_state=None):
        def _iter_folds():
            i = 0
            while i < k:
                start_idx = cut_idxs[i]
                end_idx = cut_idxs[i + 1]
                train_index = randperm[0:start_idx] + randperm[end_idx:N]
                val_index = randperm[start_idx:end_idx]
                train_data, val_data = tuple([self.examples[idx] for idx in index] for index in [train_index, val_index])
                splits = tuple(Dataset(d, self.fields)
                               for d in (train_data, val_data))

                # In case the parent sort key isn't none
                if self.sort_key:
                    for subset in splits:
                        subset.sort_key = self.sort_key
                i += 1
                yield splits

        rnd = RandomShuffler(random_state)
        N = len(self.examples)
        randperm = rnd(range(N))
        fold_len = int(N/k)
        cut_idxs = [e * fold_len for e in list(range(k))] + [N]
        data_iter = _iter_folds()
        return data_iter
