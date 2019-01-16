# Overriding torchtext methods
from gensim.models import KeyedVectors
from torchtext.data.dataset import *
from torchtext.vocab import *
import numpy as np


class MyTabularDataset(TabularDataset):
    """Subclass of torch.data.dataset.TabularDataset for k-fold cross-validation"""

    def split(self, split_ratio=0.8, stratified=False, strata_field='target',
              random_state=None):
        splits = super().split(split_ratio, stratified=stratified, strata_field=strata_field, random_state=random_state)
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
        data_iter = _iter_folds()
        return data_iter


class MyVectors(Vectors):
    #def __init__(self, *args, **kwargs):
    #    self.tokens = 0
    #    super(MyVectors, self).__init__(*args, *kwargs)
    def __init__(self, name, cache=None, to_cache = True,
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
            #self.tokens += 1
            #print(self.tokens, self.low_tokens)
            return self.vectors[self.stoi[token]]
        elif token.lower() in self.stoi:
            #self.low_tokens += 1
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
    emb_index = KeyedVectors.load_word2vec_format(path, limit = max_vectors, binary=True)
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
