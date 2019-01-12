import re
import spacy
from stop_words import GNEWS_STOP_WORDS


def lower_spacy(x):
    spacy_en = spacy.load('en')
    tokens = [tok.text.lower() for tok in spacy_en.tokenizer(x)]
    return tokens


class LowerSpacy(object):
    def __init__(self):
        self.tokenizer = spacy.load('en').tokenizer

    def __call__(self, x):
        return [tok.text.lower() for tok in self.tokenizer(x)]


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


class GNewsTokenizer(object):
    def __init__(self):
        self.spacy_en = spacy.load('en')
        self.tokenizer = self.spacy_en.tokenizer
        self.remove_all_stopwords()
        self.add_stopwords(GNEWS_STOP_WORDS)

    def __call__(self, x):
        return [tok.text.lower() for tok in self.tokenizer(x) if not tok.is_stop]

    def remove_all_stopwords(self):
        spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
        for w in spacy_stopwords:
            lexeme = self.spacy_en.vocab[w]
            lexeme.is_stop = False

    def add_stopwords(self, custom_stop_words):
        for w in custom_stop_words:
            lexeme = self.spacy_en.vocab[w]
            lexeme.is_stop = True