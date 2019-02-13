import torch
import torch.nn as nn


def section_sizes_and_lengths(lengths):
    bs = len(lengths)
    flags = lengths[1:] - lengths[:-1]
    cuts = torch.add(torch.nonzero(flags).view(-1), 1)
    cuts = torch.cat([torch.tensor([0]), cuts, torch.tensor([bs])])
    section_lengths = lengths[cuts[:-1]]
    sizes = cuts[1:] - cuts[:-1]
    return sizes, section_lengths


def max_packed(x, lengths):
    sizes, section_lengths = section_sizes_and_lengths(lengths)
    sizes = sizes.cpu().numpy().tolist()
    tensors = torch.split(x, split_size_or_sections=sizes, dim=1)
    tensors = [t[:sl] for t, sl in zip(tensors, section_lengths)]
    maxes = [torch.max(t, 0)[0] for t in tensors]
    max_tensor = torch.cat(maxes)
    return max_tensor


def mean_packed(x, lengths):
    sizes, section_lengths = section_sizes_and_lengths(lengths)
    sizes = sizes.cpu().numpy().tolist()
    tensors = torch.split(x, split_size_or_sections=sizes, dim=1)
    tensors = [t[:sl] for t, sl in zip(tensors, section_lengths)]
    means = [torch.mean(t, 0) for t in tensors]
    mean_tensor = torch.cat(means)
    return mean_tensor


def out_max_mean(x):
    out = x[-1]
    max_tensor, _ = torch.max(x, 0)
    mean_tensor = torch.mean(x, 0)
    return out, max_tensor, mean_tensor


def out_max_mean_packed(x, lengths):
    if lengths[0] == lengths[-1]:
        return out_max_mean(x)
    sizes, section_lengths = section_sizes_and_lengths(lengths)
    sizes = sizes.cpu().numpy().tolist()
    tensors = torch.split(x, split_size_or_sections=sizes, dim=1)
    tensors = [t[:sl] for t, sl in zip(tensors, section_lengths)]
    out = torch.cat([t[-1] for t in tensors])
    max_tensor = torch.cat([torch.max(t, 0)[0] for t in tensors])
    mean_tensor = torch.cat([torch.mean(t, 0) for t in tensors])
    return out, max_tensor, mean_tensor


class BiLSTM(nn.Module):
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=100, lstm_layer=2, dropout=0.2):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            dropout=dropout,
                            bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim * lstm_layer * 2, 1)
        self.cell = self.lstm

    def forward(self, sents, length):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        lstm_out, (h_n, c_n) = self.lstm(x)
        y = self.hidden2label(self.dropout(torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=1)))
        return y


class BiGRU(nn.Module):
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=100, lstm_layer=2, dropout=0.2):
        super(BiGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.gru = nn.GRU(input_size=self.embedding.embedding_dim,
                          hidden_size=hidden_dim,
                          num_layers=lstm_layer,
                          dropout=dropout,
                          bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim * lstm_layer * 2, 1)
        self.cell = self.gru

    def forward(self, sents, length):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        gru_out, h_n = self.gru(x)
        y = self.hidden2label(self.dropout(torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=1)))
        return y


class BiLSTMPoolOld(nn.Module):
    # constant length for all sequences in batch
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=100, lstm_layer=2, dropout=0.2):
        super(BiLSTMPoolOld, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            dropout=dropout,
                            bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim * 6, 1)
        self.cell = self.lstm

    def forward(self, sents, length):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        lstm_out, (h_n, c_n) = self.lstm(x)
        sl, bs, _ = lstm_out.shape
        lstm_out = lstm_out.view(sl, bs, 2 * self.hidden_dim)
        output = lstm_out[-1]
        max_pool, _ = torch.max(lstm_out, 0)
        average_pool = torch.mean(lstm_out, 0)
        y = self.hidden2label(self.dropout(torch.cat((max_pool, average_pool, output), dim=1)))
        return y


class BiLSTMPool(nn.Module):
    # varibale length for sequences in batch,  optimized for performance
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=100, lstm_layer=2, dropout=0.2):
        super(BiLSTMPool, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = lstm_layer
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            dropout=dropout,
                            bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim * 6, 1)
        self.cell = self.lstm

    def forward(self, sents, lengths=None):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths)
        lstm_out, (h_n, c_n) = self.lstm(packed_x)
        unpacked_out, unpacked_len = nn.utils.rnn.pad_packed_sequence(lstm_out)
        output, max_pool, average_pool = out_max_mean_packed(unpacked_out, unpacked_len)
        long_output = torch.cat((output, max_pool, average_pool), dim=1)
        y = self.hidden2label(self.dropout(long_output))
        return y



class BiLSTM_2FC(nn.Module):
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=100, lstm_layer=2, dropout=0.2):
        super(BiLSTM_2FC, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            dropout=dropout,
                            bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * lstm_layer * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.cell = self.lstm

    def forward(self, sents, length):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        lstm_out, (h_n, c_n) = self.lstm(x)
        y = self.fc1(self.dropout(torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=1)))
        y = self.fc2(self.dropout(y))
        return y

class BiGRUPool(nn.Module):
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=100, lstm_layer=2, dropout=0.2):
        super(BiGRUPool, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.gru = nn.GRU(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            dropout=dropout,
                            bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim * 6, 1)
        self.cell = self.gru

    def forward(self, sents, length):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        lstm_out, _ = self.gru(x)
        sl, bs, _ = lstm_out.shape
        lstm_out = lstm_out.view(sl, bs, 2 * self.hidden_dim)
        output = lstm_out[-1]
        max_pool, _ = torch.max(lstm_out, 0)
        average_pool = torch.mean(lstm_out, 0)
        y = self.hidden2label(self.dropout(torch.cat((max_pool, average_pool, output), dim=1)))
        return y


class BiGRUPool_2FC(nn.Module):
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=100, lstm_layer=2, dropout=0.2):
        super(BiGRUPool_2FC, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.gru = nn.GRU(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            dropout=dropout,
                            bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 6, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.cell = self.gru

    def forward(self, sents, length):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        lstm_out, _ = self.gru(x)
        sl, bs, _ = lstm_out.shape
        lstm_out = lstm_out.view(sl, bs, 2 * self.hidden_dim)
        output = lstm_out[-1]
        max_pool, _ = torch.max(lstm_out, 0)
        average_pool = torch.mean(lstm_out, 0)
        y = self.fc1(self.dropout(torch.cat((max_pool, average_pool, output), dim=1)))
        y = self.fc2(self.dropout(y))
        return y

class BiLSTMPool_2FC(nn.Module):
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=100, lstm_layer=2, dropout=0.2):
        super(BiLSTMPool_2FC, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.gru = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            dropout=dropout,
                            bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 6, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.cell = self.lstm

    def forward(self, sents, length):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        lstm_out, _, _ = self.lstm(x)
        sl, bs, _ = lstm_out.shape
        lstm_out = lstm_out.view(sl, bs, 2 * self.hidden_dim)
        output = lstm_out[-1]
        max_pool, _ = torch.max(lstm_out, 0)
        average_pool = torch.mean(lstm_out, 0)
        y = self.fc1(self.dropout(torch.cat((max_pool, average_pool, output), dim=1)))
        y = self.fc2(self.dropout(y))
        return y


class LinPool(nn.Module):
    # emb -> lin layer -> max, average pool -> lin layer -> label
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=100, lstm_layer=2, dropout=0.2):
        super(LinPool, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = lstm_layer
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.fc1 = nn.Linear(self.embedding.embedding_dim, self.hidden_dim)
        self.fc2 = nn.Linear(2 * self.hidden_dim + 1, 1)

    def forward(self, sents, lengths=None):
        x = self.embedding(sents)
        x1 = self.fc1(x)
        max_pool, _ = torch.max(x1, 1)
        average_pool = torch.mean(x1, 1)
        lengths = lengths.view(-1, 1).float()
        long_output = torch.cat((max_pool, average_pool, lengths), dim=1)
        y = self.fc2(self.dropout(long_output))
        return y

class LinPool4(nn.Module):
    # 4 embeddings, lin layer for each embedding -> concat all ouptus -> max, average pool -> lin layer -> label
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=100, lstm_layer=2, dropout=0.2):
        super(LinPool4, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = lstm_layer
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.fc1 = nn.Linear(300, self.hidden_dim)
        self.fc2 = nn.Linear(300, self.hidden_dim)
        self.fc3 = nn.Linear(300, self.hidden_dim)
        self.fc4 = nn.Linear(300, self.hidden_dim)
        self.fc5 = nn.Linear(2 * self.hidden_dim * self.embedding.embedding_dim//300, 1)

    def forward(self, sents, lengths=None):
        x0 = self.embedding(sents)
        x1 = self.fc1(x0[:,:,:300])
        x2 = self.fc2(x0[:,:,300:600])
        x3 = self.fc3(x0[:,:,600:900])
        x4 = self.fc4(x0[:,:,900:1200])
        x_cat = torch.cat((x1, x2, x3, x4), dim=2)
        max_pool, _ = torch.max(x_cat, 1)
        average_pool = torch.mean(x_cat, 1)
        long_output = torch.cat((max_pool, average_pool), dim=1)
        y = self.fc5(self.dropout(long_output))
        return y


class LinPool3(nn.Module):
    # 4 embeddings, lin layer for each embedding -> concat all ouptus -> max, average pool -> lin layer -> label
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=100, lstm_layer=2, dropout=0.2):
        super(LinPool3, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = lstm_layer
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.fc1 = nn.Linear(300, self.hidden_dim)
        self.fc2 = nn.Linear(300, self.hidden_dim)
        self.fc3 = nn.Linear(300, self.hidden_dim)
        self.fc4 = nn.Linear(2 * self.hidden_dim * self.embedding.embedding_dim//300, 1)

    def forward(self, sents, lengths=None):
        x0 = self.embedding(sents)
        x1 = self.fc1(x0[:,:,:300])
        x2 = self.fc2(x0[:,:,300:600])
        x3 = self.fc3(x0[:,:,600:900])
        x_cat = torch.cat((x1, x2, x3), dim=2)
        max_pool, _ = torch.max(x_cat, 1)
        average_pool = torch.mean(x_cat, 1)
        long_output = torch.cat((max_pool, average_pool), dim=1)
        y = self.fc4(self.dropout(long_output))
        return y