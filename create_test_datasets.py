import os
import pandas as pd

def read_datasets_to_df(datasets):
    dfs = []
    for d in datasets:
        dfs.append(pd.read_csv(d))
    return dfs


def dfs_to_csv(dfs, csv_names):
    if len(dfs) != len(csv_names):
        raise Exception('len(dfs) != len(csv_names)')

    for df, name in zip(dfs, csv_names):
        df.to_csv(name)
    return csv_names


def small_ds_paths(paths, new_dir, n, string):
        """
        :param paths:
        :param new_dir:
        :param n:
        :param string:
        :return: Paths for small datasets
        """
        small_paths = []

        for p in paths:
            new_path = change_dir_in_path(p, new_dir)
            root, ext = os.path.splitext(new_path)
            small_path = f'{root}_{string}{n}{ext}'
            small_paths.append(small_path)
        return small_paths


def change_dir_in_path(path, new_dir):
    _, file_name = os.path.split(path)
    new_path = os.path.join(new_dir, file_name)
    return new_path



def reduce_datasets(csv_names, new_dir, n, rand=False, seed=None):
    dfs = read_datasets_to_df(csv_names)
    exists = True
    if rand:
        seed = 42 if None else seed
        small_paths = small_ds_paths(csv_names, new_dir, n, 'rand')
        for sp in small_paths:
            if not os.path.exists(sp):
                exists = False
        if not exists:
            small_dfs = [df.sample(n) for df in dfs]
            dfs_to_csv(small_dfs, small_paths)

    else:
        small_paths = small_ds_paths(csv_names, new_dir, n, 'head')
        for sp in small_paths:
            if not os.path.exists(sp):
                exists = False
        if not exists:
            small_dfs = [df.head(n) for df in dfs]
            dfs_to_csv(small_dfs, small_paths)
    return small_paths


def reduce_embedding(emb_path, new_dir, n):
    small_emb_path = small_ds_paths([emb_path], new_dir, n, 'head')[0]
    if not os.path.exists(small_emb_path):
        with open(emb_path, 'r') as f:
            head = [next(f) for i in range(n)]
        with open(small_emb_path, 'w') as f:
            f.write(''.join(head))
    return small_emb_path

