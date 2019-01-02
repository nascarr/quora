# functions for choosing tokenizer, optimizer and model


def choose_tokenizer(tokenizer):
    if tokenizer == 'whitespace':
        return WhitespaceTokenizer()
    elif tokenizer == 'custom':
        return CustomTokenizer()
    else:
        return tokenizer