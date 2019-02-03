# Overriding torchtext methods
from gensim.models import KeyedVectors
from torchtext.data.dataset import *
from torchtext.vocab import *
import numpy as np
import itertools
from utils import print_duration
import time
import pickle


class MyTabularDataset(TabularDataset):
    """Subclass of torch.data.dataset.TabularDataset for k-fold cross-validation"""
    def __init__(self, path, format, fields, skip_header=False,
                 csv_reader_params={}, **kwargs):
        """Create a TabularDataset given a path, file format, and field list.

        Arguments:
            path (str): Path to the data file.
            format (str): The format of the data file. One of "CSV", "TSV", or
                "JSON" (case-insensitive).
            fields (list(tuple(str, Field)) or dict[str: tuple(str, Field)]:
                If using a list, the format must be CSV or TSV, and the values of the list
                should be tuples of (name, field).
                The fields should be in the same order as the columns in the CSV or TSV
                file, while tuples of (name, None) represent columns that will be ignored.

                If using a dict, the keys should be a subset of the JSON keys or CSV/TSV
                columns, and the values should be tuples of (name, field).
                Keys not present in the input dictionary are ignored.
                This allows the user to rename columns from their JSON/CSV/TSV key names
                and also enables selecting a subset of columns to load.
            skip_header (bool): Whether to skip the first line of the input file.
            csv_reader_params(dict): Parameters to pass to the csv reader.
                Only relevant when format is csv or tsv.
                See
                https://docs.python.org/3/library/csv.html#csv.reader
                for more details.
        """

        cache_path = os.path.join('.', (os.path.basename(path) + '.td'))
        try:
            with open(cache_path, 'rb') as f:
                examples = pickle.load(f)
        except:
            format = format.lower()
            make_example = {
                'json': Example.fromJSON, 'dict': Example.fromdict,
                'tsv': Example.fromCSV, 'csv': Example.fromCSV}[format]

            with io.open(os.path.expanduser(path), encoding="utf8") as f:
                if format == 'csv':
                    reader = unicode_csv_reader(f, **csv_reader_params)
                elif format == 'tsv':
                    reader = unicode_csv_reader(f, delimiter='\t', **csv_reader_params)
                else:
                    reader = f

                if format in ['csv', 'tsv'] and isinstance(fields, dict):
                    if skip_header:
                        raise ValueError('When using a dict to specify fields with a {} file,'
                                         'skip_header must be False and'
                                         'the file must have a header.'.format(format))
                    header = next(reader)
                    field_to_index = {f: header.index(f) for f in fields.keys()}
                    make_example = partial(make_example, field_to_index=field_to_index)

                if skip_header:
                    next(reader)

                examples = [make_example(line, fields) for line in reader]
                with open(cache_path, 'wb') as f:
                    pickle.dump(examples, f)

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(TabularDataset, self).__init__(examples, fields, **kwargs)

    def split(self, split_ratio=0.8, stratified=False, strata_field='target',
              random_state=None):
        splits = super().split(split_ratio, stratified=stratified, strata_field=strata_field, random_state=random_state)
        return [splits]

    def split_kfold(self, k, stratified=False, strata_field='target', is_test=False, random_state=None):
        rnd = RandomShuffler(random_state)
        if stratified:
            strata = stratify(self.examples, strata_field)
            group_generators = [self.iter_folds(group, k, rnd, is_test) for group in strata]
            for fold_data in zip(*group_generators):
                split_data = list(zip(*fold_data))
                split_data = [[e for g in data for e in g]
                                                   for data in split_data]
                splits = self.make_datasets(split_data)
                yield splits
        else:
            for fold_data in self.iter_folds(self.examples, k, rnd, is_test):
                splits = self.make_datasets(fold_data)
                yield splits

    def make_datasets(self, split_data):
        split_data = [data for data in split_data if len(data) > 0]
        percents = [sum([int(e.target) for e in data]) / len(data) * 100 for data in split_data]
        print('percent of toxic questions for train_val_test data: ', percents)
        splits = tuple(Dataset(d, self.fields) for d in split_data)
        if self.sort_key:
            for subset in splits:
                subset.sort_key = self.sort_key
        return splits

    def iter_folds(self, examples, k, rnd, is_test):
        N = len(examples)
        randperm = rnd(range(N))
        fold_len = int(N / k)
        cut_idxs = [e * fold_len for e in list(range(k))] + [N]
        i = 0
        while i < k:
            train_index, val_index, test_index = k_split_indices(randperm, cut_idxs, k, i, is_test)
            train_data, val_data, test_data = tuple([examples[idx] for idx in index]
                                                    for index in [train_index, val_index, test_index])
            i += 1
            yield train_data, val_data, test_data


def k_split_indices(randperm, cut_idxs, k, i, is_test):
    time_start = time.time()

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
    val_test_index = set(val_index + test_index)
    # train index
    print_duration(time_start, message='k_split_indices time')
    train_index = [idx for idx in randperm if idx not in val_test_index]
    print_duration(time_start, message='k_split_indices time')
    return train_index, val_index, test_index


class MyVectors(Vectors):
    # def __init__(self, *args, **kwargs):
    #    self.tokens = 0
    #    super(MyVectors, self).__init__(*args, *kwargs)
    def __init__(self, name, cache=None, to_cache=True,
                 url=None, unk_init=None, max_vectors=None):
        """
        Arguments:
           name: name of the file that contains the vectors
           cache: directory for cached vectors
           url: url for download if vectors not found in cache
           unk_init (callback): by default, initialize out-of-vocabulary word vectors
               to zero vectors; can be any function that takes in a Tensor and
               returns a Tensor of the same size
           max_vectors (int): this can be used to limit the number of
               pre-trained vectors loaded.
               Most pre-trained vector sets are sorted
               in the descending order of word frequency.
               Thus, in situations where the entire set doesn't fit in memory,
               or is not needed for another reason, passing `max_vectors`
               can limit the size of the loaded set.
         """
        cache = '.vector_cache' if cache is None else cache
        self.itos = None
        self.stoi = None
        self.vectors = None
        self.dim = None
        self.unk_init = torch.Tensor.zero_ if unk_init is None else unk_init
        if to_cache:
            self.cache(name, cache, url=url, max_vectors=max_vectors)
        else:
            self.load(name, max_vectors=max_vectors)

    def __getitem__(self, token):
        if token in self.stoi:
            # self.tokens += 1
            # print(self.tokens, self.low_tokens)
            return self.vectors[self.stoi[token]]
        elif token.lower() in self.stoi:
            # self.low_tokens += 1
            return self.vectors[self.stoi[token.lower()]]
        else:
            return self.unk_init(torch.Tensor(self.dim))

    def cache(self, name, cache, url=None, max_vectors=None):
        if os.path.isfile(name):
            path = name
            if max_vectors:
                file_suffix = '_{}.pt'.format(max_vectors)
            else:
                file_suffix = '.pt'
            path_pt = os.path.join(cache, os.path.basename(name)) + file_suffix
        else:
            path = os.path.join(cache, name)
            if max_vectors:
                file_suffix = '_{}.pt'.format(max_vectors)
            else:
                file_suffix = '.pt'
            path_pt = path + file_suffix

        if not os.path.isfile(path_pt):
            if not os.path.isfile(path) and url:
                logger.info('Downloading vectors from {}'.format(url))
                if not os.path.exists(cache):
                    os.makedirs(cache)
                dest = os.path.join(cache, os.path.basename(url))
                if not os.path.isfile(dest):
                    with tqdm(unit='B', unit_scale=True, miniters=1, desc=dest) as t:
                        try:
                            urlretrieve(url, dest, reporthook=reporthook(t))
                        except KeyboardInterrupt as e:  # remove the partial zip file
                            os.remove(dest)
                            raise e
                logger.info('Extracting vectors into {}'.format(cache))
                ext = os.path.splitext(dest)[1][1:]
                if ext == 'zip':
                    with zipfile.ZipFile(dest, "r") as zf:
                        zf.extractall(cache)
                elif ext == 'gz':
                    if dest.endswith('.tar.gz'):
                        with tarfile.open(dest, 'r:gz') as tar:
                            tar.extractall(path=cache)
            if not os.path.isfile(path):
                raise RuntimeError('no vectors found at {}'.format(path))

            logger.info("Loading vectors from {}".format(path))

            itos, vectors, dim = read_emb(path, max_vectors)

            self.itos = itos
            self.stoi = {word: i for i, word in enumerate(itos)}
            self.vectors = torch.Tensor(vectors).view(-1, dim)
            self.dim = dim
            logger.info('Saving vectors to {}'.format(path_pt))
            if not os.path.exists(cache):
                os.makedirs(cache)
            torch.save((self.itos, self.stoi, self.vectors, self.dim), path_pt)
        else:
            logger.info('Loading vectors from {}'.format(path_pt))
            self.itos, self.stoi, self.vectors, self.dim = torch.load(path_pt)

    def load(self, path, max_vectors=None):
        print('Loading embedding vectors. No cache')
        if not os.path.isfile(path):
            raise RuntimeError('no vectors found at {}'.format(path))

        logger.info("Loading vectors from {}".format(path))

        itos, vectors, dim = read_emb(path, max_vectors)

        self.itos = itos
        self.stoi = {word: i for i, word in enumerate(itos)}
        self.vectors = torch.Tensor(vectors).view(-1, dim)
        self.dim = dim


def read_emb(path, max_vectors):
    ext = os.path.splitext(path)[1][1:]
    if ext == 'bin':
        itos, vectors, dim = emb_from_bin(path, max_vectors)
    else:
        itos, vectors, dim = emb_from_txt(path, ext, max_vectors)
    return itos, vectors, dim


def emb_from_txt(path, ext, max_vectors):
    if ext == 'gz':
        open_file = gzip.open
    else:
        open_file = open

    vectors_loaded = 0
    with open_file(path, 'rb') as f:
        num_lines, dim = _infer_shape(f)
        if not max_vectors or max_vectors > num_lines:
            max_vectors = num_lines

        itos, vectors, dim = [], torch.zeros((max_vectors, dim)), None

        for line in f:
            # Explicitly splitting on " " is important, so we don't
            # get rid of Unicode non-breaking spaces in the vectors.
            entries = line.rstrip().split(b" ")

            word, entries = entries[0], entries[1:]
            if dim is None and len(entries) > 1:
                dim = len(entries)
            elif len(entries) == 1:
                logger.warning("Skipping token {} with 1-dimensional "
                               "vector {}; likely a header".format(word, entries))
                continue
            elif dim != len(entries):
                raise RuntimeError(
                    "Vector for token {} has {} dimensions, but previously "
                    "read vectors have {} dimensions. All vectors must have "
                    "the same number of dimensions.".format(word, len(entries),
                                                            dim))

            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except UnicodeDecodeError:
                logger.info("Skipping non-UTF8 token {}".format(repr(word)))
                continue

            vectors[vectors_loaded] = torch.tensor([float(x) for x in entries])
            vectors_loaded += 1
            itos.append(word)

            if vectors_loaded == max_vectors:
                break

    return itos, vectors, dim


def emb_from_bin(path, max_vectors):
    emb_index = KeyedVectors.load_word2vec_format(path, limit=max_vectors, binary=True)
    itos = emb_index.index2word
    vectors = emb_index.vectors
    dim = emb_index.vector_size
    return itos, vectors, dim


def _infer_shape(f):
    num_lines, vector_dim = 0, None
    for line in f:
        if vector_dim is None:
            row = line.rstrip().split(b" ")
            vector = row[1:]
            # Assuming word, [vector] format
            if len(vector) > 2:
                # The header present in some (w2v) formats contains two elements.
                vector_dim = len(vector)
                num_lines += 1  # First element read
        else:
            num_lines += 1
    f.seek(0)
    return num_lines, vector_dim
