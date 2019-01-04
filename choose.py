# functions for choosing tokenizer, optimizer and model
from tokenizers import *
from models import *
import torch.optim as optim


def choose_tokenizer(tokenizer):
    if tokenizer == 'whitespace':
        return WhitespaceTokenizer()
    elif tokenizer == 'custom':
        return CustomTokenizer()
    elif tokenizer == 'lowerspacy':
        return LowerSpacy()
    else:
        return tokenizer


def choose_model(text, args):
    if args.model == 'BiLSTM':
        model = BiLSTM(text.vocab.vectors,
                       lstm_layer=args.n_layers,
                       padding_idx=text.vocab.stoi[text.pad_token],
                       hidden_dim=args.hidden_dim,
                       dropout=args.dropout).cuda()
    if args.model == 'BiGRU':
        model = BiGRU(text.vocab.vectors,
                      lstm_layer=args.n_layers,
                      padding_idx=text.vocab.stoi[text.pad_token],
                      hidden_dim=args.hidden_dim,
                      dropout=args.dropout).cuda()
    if args.model == 'BiLSTMPool':
        model = BiLSTMPool(text.vocab.vectors,
                           lstm_layer=args.n_layers,
                           padding_idx=text.vocab.stoi[text.pad_token],
                           hidden_dim=args.hidden_dim,
                           dropout=args.dropout).cuda()
    if args.model == 'BiGRUPool':
        model = BiGRUPool(text.vocab.vectors,
                          lstm_layer=args.n_layers,
                          padding_idx=text.vocab.stoi[text.pad_token],
                          hidden_dim=args.hidden_dim,
                          dropout=args.dropout).cuda()
    if args.model == 'BiLSTM_2FC':
        model = BiLSTM_2FC(text.vocab.vectors,
                           lstm_layer=args.n_layers,
                           padding_idx=text.vocab.stoi[text.pad_token],
                           hidden_dim=args.hidden_dim,
                           dropout=args.dropout).cuda()

    if args.model == 'BiGRUPool_2FC':
        model = BiGRUPool_2FC(text.vocab.vectors,
                              lstm_layer=args.n_layers,
                              padding_idx=text.vocab.stoi[text.pad_token],
                              hidden_dim=args.hidden_dim,
                              dropout=args.dropout).cuda()

    if args.model == 'BiLSTMPool_2FC':
        model = BiLSTMPool_2FC(text.vocab.vectors,
                               lstm_layer=args.n_layers,
                               padding_idx=text.vocab.stoi[text.pad_token],
                               hidden_dim=args.hidden_dim,
                               dropout=args.dropout).cuda()
    return model


def choose_optimizer(params, args):
    if args.optim == 'Adam':
        optimizer = optim.Adam(params, lr=args.lr)
    elif args.optim == 'AdamW':
        optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.99))
    return optimizer