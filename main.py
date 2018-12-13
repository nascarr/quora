import os
import random
import torch.nn as nn
import torch.optim as optim
import argparse

from models import BiLSTM
from preprocess import preprocess, iterate
from learner import Learner
from utils import submit
from test.create_test_datasets import reduce_embedding, reduce_datasets


def main(args, train_csv, test_csv, embedding, cache):
    train, test, text, qid = preprocess(train_csv, test_csv, args.tokenizer, embedding, cache)
    random.seed(args.seed)
    train, val = train.split(split_ratio=args.split_ratio, random_state=random.getstate())
    train_iter, val_iter, test_iter = iterate(train, val, test, args.batch_size)

    eval_every = len(list(iter(train_iter)))/args.n_eval
    if args.model == 'BiLSTM':
        model = BiLSTM(text.vocab.vectors, lstm_layer=args.n_layers, padding_idx=text.vocab.stoi[text.pad_token], hidden_dim=args.hidden_dim, dropout=args.dropout).cuda()
    # loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_w]).cuda())
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    dataloaders = train_iter, val_iter, test_iter
    learn = Learner(model, dataloaders, loss_function, optimizer)
    learn.fit(args.epoch, eval_every, args.f1_tresh, args.early_stop, args.warmup_epoch)

    # predict test labels
    learn.load()
    test_label, _, test_ids = learn.predict_labels(is_test=True, tresh=[0.01, 0.5, 0.01])
    test_ids = [qid.vocab.itos[i] for i in test_ids]
    submit(test_ids, test_label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--machine', default='dt', choices=['dt', 'kaggle'])
    arg('--mode', default='run', choices=['test', 'run'])
    arg('--epoch', '-e', default=7, type=int)
    arg('--lr','-lr', default=1e-3, type=float)
    arg('--batch_size', '-bs', default=512, type=int)
    arg('--n_eval', '-ne', default=1, type=int, help='Number of validation set evaluations during 1 epoch')
    arg('--warmup_epoch', '-we', default=2, type=int, help='Number of epochs without fine tuning')
    arg('--early_stop', '-es', default=1, type=int, help='Stop training if no improvement during this number of epochs')
    arg('--split_ratio', '-sr', default=0.8, type=float)
    arg('--seed', default=2018, type=int)
    arg('--tokenizer', '-t', default='spacy', choices=['spacy'])
    arg('--embedding', '-em', default='glove', choices=['glove', 'google_news','paragram', 'wiki_news'])
    arg('--f1_tresh', '-ft', default=0.33, type=float)

    #model params
    arg('--model', '-m', default = 'BiLSTM', choices=['BiLSTM'])
    arg('--n_layers', '-l', default=2, help='Number of layers in model')
    arg('--hidden_dim', '-hd', default=100)
    arg('--dropout', '-d', default=0.2)

    args = parser.parse_args(args=[])
    print(args)

    if args.embedding == 'glove':
        emb_path = 'embeddings/glove.840B.300d/glove.840B.300d.txt'
    # TODO: add other embeddings

    if args.machine == 'dt':
        data_dir = cache = './data'
    elif args.machine == 'kaggle':
        data_dir = '../input'
        cache = '.'

    train_csv = os.path.join(data_dir, 'train.csv')
    test_csv = os.path.join(data_dir, 'test.csv')
    emb_path = os.path.join(data_dir, emb_path)

    if args.mode == 'test':
        # create smaller files for testing main function
        n_cut = 1000
        n_cut_emb = 10000
        args.batch_size = n_cut/5
        args.n_eval = 1

        if args.machine == 'kaggle':
            data_dir = '.'

        train_small_csv, test_small_csv = reduce_datasets([train_csv, test_csv], data_dir, n_cut)
        emb_small_path = reduce_embedding(emb_path, data_dir, n_cut_emb)
        train_csv, test_csv, emb_path = train_small_csv, test_small_csv, emb_small_path
    elif args.mode == 'run':
        pass  # all parameters for 'run' mode defined above

    main(args, train_csv, test_csv, emb_path, cache)
