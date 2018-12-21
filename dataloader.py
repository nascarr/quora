from torchtext.data.dataset import *
import numpy as np


class MyTabularDataset(TabularDataset):
    """Subclass of torch.data.dataset.TabularDataset for k-fold cross-validation"""
    def split(self, split_ratio=0.8, stratified=False, strata_field='label',
              random_state=None):
        splits = super().split(split_ratio, stratified, strata_field, random_state)
        return [splits]

    def split_kfold(self, k, stratified=False, is_test=False, random_state=None):
        def _iter_folds():
            i = 0
            while i < k:
                # val index
                val_start_idx = cut_idxs[i]
                val_end_idx = cut_idxs[i + 1]
                val_index = randperm[val_start_idx:val_end_idx]

                # test index
                if is_test:
                    if i <= k - 2:
                        test_start_idx = cut_idxs[i + 1]
                        test_end_idx = cut_idxs[i + 2]
                    else:
                        test_start_idx = cut_idxs[0]
                        test_end_idx = cut_idxs[1]
                    test_index = randperm[test_start_idx:test_end_idx]
                else:
                    test_index = []

                # train index
                train_index = list(set(randperm) - set(test_index) - set(val_index))

                # split examples by index and create datasets
                train_data, val_data, test_data = tuple([self.examples[idx] for idx in index]
                                                        for index in [train_index, val_index, test_index])
                splits = tuple(Dataset(d, self.fields)
                               for d in (train_data, val_data, test_data) if d)

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
        print(cut_idxs)
        data_iter = _iter_folds()
        return data_iter
