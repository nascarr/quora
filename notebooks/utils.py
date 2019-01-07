import re
import numpy as np

def print_df_values(df):
    for q in df.values:
        print(q)
        print('\n')


def token_info(tokens):
    print('Number of tokens: ', len(tokens))
    print('First 10 tokens: ', tokens[:10])


def txtlen(path, header=False):
    count = 0
    with open(path) as f:
        for line in f:
            count += 1
    if header:
        count -= 1
    return count


def emb_to_array(path, max_n=None):
    total = txtlen(path)
    if max_n:
        if max_n < total:
            total = max_n
    emb_vectors = np.zeros((total, 300))
    idx = 0
    for t, e in read_emb(path):
        if idx >= total:
            break
        e = np.array(e)
        emb_vectors[idx] = e
        idx += 1
    return emb_vectors


def read_emb(path):
    with open(path) as f:
        for line in f:
            cut = line.find(' ')
            tok = line[:cut]
            emb = line[cut + 1:].split()
            emb = [float(e) for e in emb]
            yield tok, emb


class TokenTypes:
    def __init__(self):
        self.num_tokens = []
        self.symbol_tokens = []
        self.low_tokens = []
        self.up_tokens = []
        self.caps_tokens = []

    def __call__(self, tokens):
        num_re = re.compile('[0-9]')
        low_re = re.compile('[a-z]')
        up_re = re.compile('[A-Z]')
        caps_re = re.compile('^[A-Z]*$')
        letnum_re = re.compile('[a-zA-Z0-9]')
        for t in tokens:
            if num_re.search(t):
                self.num_tokens.append(t)
            if not letnum_re.search(t):
                self.symbol_tokens.append(t)
            if low_re.match(t):
                self.low_tokens.append(t)
            if up_re.match(t):
                self.up_tokens.append(t)
            if caps_re.match(t):
                self.caps_tokens.append(t)
        print('Number of num tokens: ', len(self.num_tokens))
        print('Number of symbol tokens: ', len(self.symbol_tokens))
        print('Number of lowcase tokens: ', len(self.low_tokens))
        print('Number of upper case tokens: ', len(self.up_tokens))
        print('Nubmer of capslock tokens: ', len(self.caps_tokens))

        print('\n')

        print('Freq num tokens: ', self.num_tokens[:10])
        print('Rare num tokens: ', self.num_tokens[-10:], '\n')
        print('Freq symbol tokens: ', self.symbol_tokens[:10])
        print('Rare symbol tokens: ', self.symbol_tokens[-10:], '\n')
        print('Freq low tokens: ', self.low_tokens[:10])
        print('Rare low tokens: ', self.low_tokens[-10:], '\n')
        print('Freq up tokens: ', self.up_tokens[:10])
        print('Rare up tokens: ', self.up_tokens[-10:], '\n')
        print('Freq caps tokens: ', self.caps_tokens[:10])
        print('Rare caps tokens: ', self.caps_tokens[-10:])




