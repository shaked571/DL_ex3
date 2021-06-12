from typing import Tuple, List

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from vocab import Vocab, CharsVocab, SubWords
import abc


class BiLSTM(nn.Module, abc.ABC):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab: Vocab, dropout=0.2, sent_len=128):
        super(BiLSTM, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.vocab_size = self.vocab.vocab_size
        self.sent_len = sent_len
        self.embed_dim = 50
        self.dropout_val = dropout
        self.embedding = self.get_embedding_layer()

        self.blstm = nn.LSTM(input_size=self.embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=2,
                            bidirectional=True)
        self.linear = nn.Linear(2*hidden_dim, self.vocab.num_of_labels)


    def forward(self, x, x_lens):
        embeds = self.get_embed_vectors(x, x_lens)
        x_packed = pack_padded_sequence(embeds, x_lens, batch_first=True, enforce_sorted=False)
        out, (last_hidden_state, c_n) = self.blstm(x_packed)
        out2, _ = pad_packed_sequence(out, batch_first=True)
        out3 = self.linear(out2)
        out4 = out3.flatten(0, 1)
        return out4

    def load_model(self, path):
        checkpoint = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(checkpoint)

    @abc.abstractmethod
    def get_embedding_layer(self):
        pass

    @abc.abstractmethod
    def get_embed_vectors(self, x, x_lens):
        pass


class BiLSTMVanila(BiLSTM):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab: Vocab, dropout=0.2, sent_len=128):
        super().__init__(embedding_dim, hidden_dim, vocab, dropout, sent_len)

    def get_embedding_layer(self):
        return nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)

    def get_embed_vectors(self, x, x_lens):
        return self.embedding(x)


class LSTMEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(LSTMEmbedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.embed.embedding_dim, hidden_size=hidden_dim)


    def forward(self, x, x_lens):
        out = self.embed(x)
        out = pack_padded_sequence(out, x_lens, batch_first=True, enforce_sorted=False)
        _, (ht, c_n) = self.lstm(out)

        return ht


class BiLSTMChar(BiLSTM):
    def __init__(self, embedding_dim: int, hidden_dim: int, lstm_hidden_dim: int, vocab: Vocab, chars_vocab: CharsVocab, dropout=0.2, sent_len=128):
        self.lstm_hidden_dim = lstm_hidden_dim
        self.char_vocab = chars_vocab
        super().__init__(embedding_dim, hidden_dim, vocab, dropout, sent_len)
        self.blstm = nn.LSTM(input_size=self.lstm_hidden_dim,
                            hidden_size=hidden_dim,
                            num_layers=2,
                            bidirectional=True)


    def get_embedding_layer(self):
        return LSTMEmbedding(self.char_vocab.vocab_size, self.embed_dim, self.lstm_hidden_dim)

    def get_embed_vectors(self, x, x_lens):
        embed_char, lens = self.transform_embed_char(x)
        ht = self.embedding(embed_char, lens)
        split_words = torch.split(ht[-1], x_lens, dim=0)
        res = torch.nn.utils.rnn.pad_sequence(split_words, batch_first=True)
        return res



    def transform_embed_char(self, x):
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
        chars_tensor = torch.tensor(chars_tensor).to(torch.int64).to(self.device)
        return chars_tensor, lens


class BiLSTMSubWords(BiLSTM):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab: Vocab, sub_words: SubWords, dropout=0.2, sent_len=128):
        super().__init__(embedding_dim, hidden_dim, vocab, dropout, sent_len)
        self.sub_words = sub_words
        self.prefix_embedding = nn.Embedding(self.sub_words.prefix_num, self.embed_dim, padding_idx=0)
        self.suffix_embedding = nn.Embedding(self.sub_words.suffix_num, self.embed_dim, padding_idx=0)

    def get_embedding_layer(self):

        return nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)

    def get_sub_words_tensor(self, words):
        words_prefixes = []
        words_suffixes = []
        for w in words:
            prefix, suffix = self.sub_words.get_sub_words_indexes_by_word(w)
            words_prefixes.append(prefix)
            words_suffixes.append(suffix)
        prefixes_tensor = torch.tensor(words_prefixes).to(torch.int64)
        suffixes_tensor = torch.tensor(words_suffixes).to(torch.int64)

        prefixes_tensor = prefixes_tensor.to(self.device)
        suffixes_tensor = suffixes_tensor.to(self.device)
        return prefixes_tensor, suffixes_tensor

    def get_embed_vectors(self, x, x_lens):
        batch = None
        for sent_tensor in x:
            words = [self.vocab.i2token[i.item()] for i in sent_tensor]
            # words_tensor = torch.tensor([self.vocab.get_word_index(word) for word in words]).to(torch.int64)
            prefixes_tensor, suffixes_tensor = self.get_sub_words_tensor(words)
            out_word = self.embedding(sent_tensor)
            out_pre = self.prefix_embedding(prefixes_tensor)
            out_suf = self.suffix_embedding(suffixes_tensor)
            embeds = torch.stack((out_word, out_pre, out_suf), dim=0).sum(axis=0)
            if batch is None:
                batch = torch.unsqueeze(embeds, dim=0)
            else:
                batch = torch.cat((batch, torch.unsqueeze(embeds, dim=0)), dim=0)
        return batch


class BiLSTMConcat(BiLSTM):
    def __init__(self, embedding_dim: int, hidden_dim: int, lstm_hidden_dim: int, vocab: Vocab, chars_vocab: CharsVocab, dropout=0.2, sent_len=128):
        super().__init__(embedding_dim, hidden_dim, vocab, dropout, sent_len)

        self.embedding_bilstm = BiLSTMVanila(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab=vocab,
                                             dropout=dropout, sent_len=sent_len)
        self.chars_bilstm = BiLSTMChar(embedding_dim=embedding_dim, hidden_dim=hidden_dim, lstm_hidden_dim=lstm_hidden_dim,
                                       vocab=vocab, chars_vocab=chars_vocab, dropout=dropout, sent_len=sent_len)
        self.blstm = nn.LSTM(input_size=self.embed_dim + lstm_hidden_dim,
                            hidden_size=hidden_dim,
                            num_layers=2,
                            bidirectional=True)


    def get_embedding_layer(self):
        return nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)

    def get_embed_vectors(self, x, x_lens):
        embed_vec = self.embedding_bilstm.get_embed_vectors(x, x_lens)
        chars_vec = self.chars_bilstm.get_embed_vectors(x, x_lens)
        return torch.cat((embed_vec, chars_vec), dim=2)


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

