from typing import Tuple, List

import torch
from torch import nn, Tensor
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



class LSTMEmbedding(nn.Module):
    def __init__(self,vocab_size, embed_dim,hidden_dim, dropout_val):
        super(LSTMEmbedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.embed.embedding_dim, hidden_size=hidden_dim, dropout=dropout_val)

    def forward(self, x, x_lens):
        out = self.embed(x)
        out = pack_padded_sequence(out, x_lens, batch_first=True, enforce_sorted=False)
        out = self.lstm(out)
        return out





class BiLSTMChar(BiLSTM):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab: Vocab,chars_vocab: CharsVocab, dropout=0.2, sent_len=128):
        super().__init__(embedding_dim, hidden_dim, vocab, dropout, sent_len)
        self.char_vocab = chars_vocab


    # def LSTMEmbedding(self,x, lens ):
    #     embed = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
    #     lstm = nn.LSTM(input_size=embed.embedding_dim,
    #                         hidden_size=self.hidden_dim,
    #                         dropout=self.dropout_val)
    #
    #
    def get_embedding(self):
        return LSTMEmbedding(self.vocab_size, self.embed_dim,self.hidden_dim, self.dropout_val)

    def forward(self, x, x_lens):
        embed_char, lens = self.tarnsform_embded_char(x)
        _, (last_hidden_state, c_n) = self.embedding(embed_char, lens)

        embeds = c_n[-1]
        #re pack them
        x_packed = pack_padded_sequence(embeds, x_lens, batch_first=True, enforce_sorted=False)
        out, (last_hidden_state, c_n) = self.blstm(x_packed)
        out, _ = pad_packed_sequence(out, total_length=self.sent_len, batch_first=True)
        out = self.relu(out)
        out = self.linear(out)
        out = out[:, :max(x_lens)]

        return out.flatten(0, 1)

    def repack(self, x, x_lens):
        max_len = max(x_lens)
        new_x = []
        first = 0
        for sent_len in x_lens:
            tensor_sent = x[first:sent_len + first]
            first += sent_len

    def tarnsform_embded_char(self, x):
        sents = []
        max_len_word = -1
        for s in x:
            words = [self.vocab.i2token[i.item()] for i in s if i.item() != self.vocab.PAD_IDX]
            sents.append(words)
            for w in words:
                if len(w) > max_len_word:
                    max_len_word = len(w)
        res = []
        lens = []
        for s in sents:
            chars_tensor, l = self.get_chars_tensor(s, max_len_word)
            res += chars_tensor
            lens += l
        res = torch.stack(res)
        return res, lens


    def get_chars_tensor(self, words, max_len) -> Tuple[Tensor, List[int]]:
        chars_tensor = []
        lens = []
        for word in words:
            chars_indices = self.char_vocab.get_chars_indexes_by_word(word)
            chars_tensor.append(chars_indices)
        for c_w in chars_tensor:
            lens.append(len(c_w))
            if len(c_w) < max_len:
                c_w += [0] * (max_len - len(c_w))
        chars_tensor = torch.tensor(chars_tensor).to(torch.int64)
        return chars_tensor, lens


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

