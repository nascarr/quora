import torch
import torch.nn as nn



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

    def forward(self, sents):
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

    def forward(self, sents):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        gru_out, h_n = self.gru(x)
        y = self.hidden2label(self.dropout(torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=1)))
        return y

class BiLSTMPool(nn.Module):
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=100, lstm_layer=2, dropout=0.2):
        super(BiLSTMPool, self).__init__()
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

    def forward(self, sents, lengths=None):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths)
        lstm_out, (h_n, c_n) = self.lstm(packed_x)
        unpacked_out, unpacked_len = nn.utils.rnn.pad_packed_sequence(lstm_out)
        sl, bs, _ = unpacked_out.shape
        lstm_out = unpacked_out
        output_list = [lstm_out[:l, i, :] for i, l in enumerate(unpacked_len.cpu().numpy())]
        output = torch.stack([t[-1, :] for t in output_list], dim=1)
        max_pool= torch.stack([torch.max(t, 0)[0] for t in output_list], dim=1)
        average_pool = torch.stack([torch.mean(t, 0) for t in output_list], dim=1)
        long_output = torch.cat((output, max_pool, average_pool), dim=0)
        long_output = torch.transpose(long_output, dim0=1, dim1=0)
        y = self.hidden2label(self.dropout(long_output))
        return y


class BiLSTMPool_fast(nn.Module):
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=100, lstm_layer=2, dropout=0.2):
        super(BiLSTMPool_fast, self).__init__()
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
        sl, bs, _ = unpacked_out.shape
        output = h_n.view(self.num_layers, 2, bs, self.hidden_dim)[1]
        output = torch.cat((output[0], output[1]), dim=1)
        max_pool, _ = torch.max(unpacked_out, 0)
        average_pool = torch.mean(unpacked_out, 0)
        long_output = torch.cat((output, max_pool, average_pool), dim=1)
        #long_output = torch.transpose(long_output, dim0=1, dim1=0)
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

    def forward(self, sents):
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

    def forward(self, sents):
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

    def forward(self, sents):
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

    def forward(self, sents):
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