# functions for choosing tokenizer, optimizer and model
import models
import torch.optim as optim


def choose_model(model_name, text, n_layers, hidden_dim, dropout):
    model = getattr(models, model_name)(text.vocab.vectors,
                       lstm_layer=n_layers,
                       padding_idx=text.vocab.stoi[text.pad_token],
                       hidden_dim=hidden_dim,
                       dropout=dropout).cuda()
    return model


def choose_optimizer(params, args):
    if args.optim == 'Adam':
        optimizer = optim.Adam(params, lr=args.lr)
    elif args.optim == 'AdamW':
        optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.99))
    return optimizer