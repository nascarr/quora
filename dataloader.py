from torchtext.data.dataset import *

class MyTabularDataset(TabularDataset):
    """Subclass of torch.data.dataset.TabularDataset for k-fold cross-validation"""
    def split_kfold(self, k, stratified=False, test=False, random_state=None):
        def _iter_folds():
            i = 0
            while i < k:
                val_index = indices[i]
                train_index = [idx for tr_index in ]
                data = tuple([self.examples[i] for i in index] for index in indices)
                yield data

        rnd = RandomShuffler(random_state)
        N = len(self.examples)
        randperm = rnd(range(N))
        fold_len = int(N/k)
        indices = []
        for i in range(k):
            if i < k - 1:
                fold_indices = randperm[fold_len * i: fold_len * (i + 1)]
            else:
            # i == k - 1 (last fold)
                fold_indices = randperm[fold_len * i: N]
            indices.append(fold_indices)
        return _iter_folds()
