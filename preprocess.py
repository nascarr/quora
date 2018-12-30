import torchtext.data as data
import torchtext.vocab as vocab
import time
import re

from utils import print_duration
from dataloader import MyTabularDataset


def choose_tokenizer(tokenizer):
    if tokenizer == 'whitespace':
        return WhitespaceTokenizer()
    elif tokenizer == 'custom':
        return CustomTokenizer()
    else:
        return tokenizer


class WhitespaceTokenizer(object):
    def __init__(self):
        pass

    def __call__(self, text):
        words = text.split(' ')
        return words


class CustomTokenizer(object):
    def __init__(self):
        #self.allowed_re = re.compile('^[A-Za-z0-9.,?!()]*$')
        self.punct = re.compile('[,!.?()]')

    def __call__(self, text):
        substrings = text.split(' ')
        tokens = []
        for ss in substrings:
            punct_match = self.punct.search(ss)
            if punct_match:
                start = punct_match.start()
                ss_tokens = [ss[:start], ss[start], ss[start + 1:]]
                ss_tokens = [t for t in ss_tokens if len(t) > 0]
            else:
                ss_tokens = [ss]
            tokens.extend(ss_tokens)
        return tokens


def preprocess(train_csv, test_csv, tokenizer, embeddings, cache):
        # types of csv columns
        location = './cachedir'
        time_start = time.time()
        text = data.Field(batch_first=True, tokenize=tokenizer)
        qid = data.Field()
        target = data.Field(sequential=False, use_vocab=False, is_target=True)

        # read and tokenize data
        print('Reading data.')
        train = MyTabularDataset(path=train_csv, format='csv',
                                    fields={'qid': ('qid', qid),
                                            'question_text': ('text', text),
                                            'target': ('target', target)})
        test = MyTabularDataset(path=test_csv, format='csv',
                                   fields={'qid': ('qid', qid),
                                           'question_text': ('text', text)})
        text.build_vocab(train, test, min_freq=1)
        qid.build_vocab(train, test)
        print_duration(time_start, 'Time to read and tokenize data: ')

        # embeddings lookup
        print('Embedding lookup...')
        time_start = time.time()
        text.vocab.load_vectors(vocab.Vectors(embeddings, cache=cache))
        print_duration(time_start, 'Time for embedding lookup: ')

        return train, test, text, qid


def iterate(train, val, test, batch_size):
        train_iter = data.BucketIterator(dataset=train,
                                         batch_size=batch_size,
                                         sort_key=lambda x: x.text.__len__(),
                                         shuffle=True,
                                         sort=False)

        val_iter = data.BucketIterator(dataset=val,
                                       batch_size=batch_size,
                                       sort_key=lambda x: x.text.__len__(),
                                       train=False,
                                       sort=False)

        test_iter = data.BucketIterator(dataset=test,
                                        batch_size=batch_size,
                                        sort_key=lambda x: x.text.__len__(),
                                        sort=False,
                                        train=False)
        return train_iter, val_iter, test_iter

if __name__ == '__main__':
    tok = CustomTokenizer()
    str = 'I love yo-u! Yes, ,no, yes?'
    print(tok(str))