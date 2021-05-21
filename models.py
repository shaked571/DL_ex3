import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from vocab import Vocab


class BiLSTMVanila(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab: Vocab, dropout=0.2):
        super(BiLSTMVanila, self).__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.vocab_size = self.vocab.vocab_size
        self.embed_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)

        self.blstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=2,
                            dropout=dropout,
                            bidirectional=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_dim, self.vocab.num_of_labels)

    def forward(self, x, x_lens):
        embeds = self.embedding(x)
        x_packed = pack_padded_sequence(embeds, x_lens, batch_first=True, enforce_sorted=False)
        out, (last_hidden_state, c_n) = self.blstm(x_packed)
        out, _ = pad_packed_sequence(out)
        out = self.relu(out)
        out = self.linear(out)

        return out

    def load_model(self, path):
        checkpoint = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(checkpoint)




# class MLPSubWords(MLP):
#     def __init__(self, embedding_size: int, hidden_dim: int, vocab: Vocab, sub_words: SubWords):
#         super().__init__(embedding_size, hidden_dim, vocab)
#         self.sub_words = sub_words
#         self.prefix_embedding = nn.Embedding(self.sub_words.prefix_num, self.embed_dim)
#         self.suffix_embedding = nn.Embedding(self.sub_words.suffix_num, self.embed_dim)
#
#     def forward(self, x):
#         out_word = self.embedding(x[torch.arange(x.size(0)), 0])
#         out_pre = self.prefix_embedding(x[torch.arange(x.size(0)), 1])
#         out_suf = self.suffix_embedding(x[torch.arange(x.size(0)), 2])
#         out = torch.stack((out_word, out_pre, out_suf), dim=0).sum(axis=0)
#         out = out.view(out.size(0), -1)
#         out = self.linear1(out)
#         out = self.tanh(out)
#         out = self.linear2(out)
#         return out


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

