import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from vocab import Vocab, CharsVocab
import abc


class BiLSTM(nn.Module, abc.ABC):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab: Vocab, dropout=0.2, sent_len=128):
        super(BiLSTM, self).__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.vocab_size = self.vocab.vocab_size
        self.sent_len = sent_len
        self.embed_dim = embedding_dim
        self.dropout_val = dropout
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = self.get_embedding()

        self.blstm = nn.LSTM(input_size=self.embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=2,
                            dropout=dropout,
                            bidirectional=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(2*hidden_dim, self.vocab.num_of_labels)

    def forward(self, x, x_lens):
        embeds = self.embedding(x)
        x_packed = pack_padded_sequence(embeds, x_lens, batch_first=True, enforce_sorted=False)
        out, (last_hidden_state, c_n) = self.blstm(x_packed)
        out, _ = pad_packed_sequence(out, total_length=self.sent_len, batch_first=True)
        out = self.relu(out)
        out = self.linear(out)
        out = out[:, :max(x_lens)]

        return out.flatten(0, 1)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(checkpoint)

    @abc.abstractmethod
    def get_embedding(self):
        pass


class BiLSTMVanila(BiLSTM):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab: Vocab, dropout=0.2, sent_len=128):
        super().__init__(embedding_dim, hidden_dim, vocab, dropout, sent_len)

    def get_embedding(self):
        return nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)


class BiLSTMChar(BiLSTM):
    def __init__(self, embedding_dim: int, hidden_dim: int, chars_vocab: CharsVocab, dropout=0.2, sent_len=128):
        super().__init__(embedding_dim, hidden_dim, chars_vocab, dropout, sent_len)

    def get_embedding(self):
        embed = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        lstm = nn.LSTM(input_size=embed.embedding_dim,
                            hidden_size=self.hidden_dim,
                            dropout=self.dropout_val)
        seq = nn.Sequential(embed, lstm)
        return seq

    def forward(self, x, x_lens):
        _, (last_hidden_state, c_n) = self.embedding(x)
        embeds = c_n[-1]
        x_packed = pack_padded_sequence(embeds, x_lens, batch_first=True, enforce_sorted=False)
        out, (last_hidden_state, c_n) = self.blstm(x_packed)
        out, _ = pad_packed_sequence(out, total_length=self.sent_len, batch_first=True)
        out = self.relu(out)
        out = self.linear(out)
        out = out[:, :max(x_lens)]

        return out.flatten(0, 1)


class SeqLstm(nn.Module):

    def __init__(self, vocab: Vocab, embedding_dim=50, hidden_dim=100):
        super(SeqLstm, self).__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab.vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim)
        self.linear1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(self.hidden_dim, self.vocab.num_of_labels)

    def forward(self, x, x_lens):
        embeds = self.word_embeddings(x)
        x_packed = pack_padded_sequence(embeds, x_lens, batch_first=True, enforce_sorted=False)
        out, (ht, ct) = self.lstm(x_packed)
        out = self.linear1(ht[-1])
        out = self.tanh(out)
        out = self.linear2(out)
        return out

