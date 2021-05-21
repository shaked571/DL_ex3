import os
from dataclasses import dataclass
from typing import List, Optional
from torch.utils.data import DataLoader, Dataset
import torch
from vocab import Vocab


@dataclass
class InputExample:
    """
    A single training/test example for token classification.
    Args:
        guid:  Unique id for the example.
        words: list. The words of the sequence.
        label: (Optional) str. The label of the middle word in the window
    """
    guid: str
    words: List[str]
    labels: Optional[List[str]]


class PreProcessor(object):
    pass


class TokenDataFile(Dataset):
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

    def __init__(self, task: str, data: str, vocab: Vocab, partial=None):
                 # sub_words: SubWords = None,
                 # char_vocab: CharsVocab = None):
        self.data_path = os.path.join(self.BASE_PATH, data)
        self.separator = " " if task == "pos" else "\t"
        self.vocab = vocab
        # self.sub_words = sub_words
        # self.char_vocab = char_vocab
        self.data, self.labels = self.get_examples_and_labels()
        if partial == 'train':
            self.data, self.labels = self.data[:int(len(self.data) * 0.9)], self.labels[:int(len(self.data) * 0.9)]
        elif partial == 'dev':
            self.data, self.labels = self.data[int(len(self.data) * 0.9):], self.labels[int(len(self.data) * 0.9):]


    def get_examples_and_labels(self):
        examples = []
        labels = []
        with open(self.data_path, mode="r") as f:
            lines = f.readlines()
        for line in lines:
            example, label = line.strip().split(self.separator)
            examples.append(example)
            labels.append(label)

        return examples, labels

    def __len__(self):
        return len(self.data)

    # def get_sub_words_tensor(self, words):
    #     words_prefixes = []
    #     words_suffixes = []
    #     for w in words:
    #         prefix, suffix = self.sub_words.get_sub_words_indexes_by_word(w)
    #         words_prefixes.append(prefix)
    #         words_suffixes.append(suffix)
    #     prefixes_tensor = torch.tensor(words_prefixes).to(torch.int64)
    #     suffixes_tensor = torch.tensor(words_suffixes).to(torch.int64)
    #     return prefixes_tensor, suffixes_tensor

    def __getitem__(self, index):
        word = self.data[index]
        label = self.labels[index]
        words_tensor = torch.tensor([self.vocab.get_word_index(word)]).to(torch.int64)
        label_tensor = torch.tensor([self.vocab.label2i[label]]).to(torch.int64)

        # if self.sub_words:
        #     prefixes_tensor, suffixes_tensor = self.get_sub_words_tensor(words)
        #     words_tensor = torch.stack((words_tensor, prefixes_tensor, suffixes_tensor), dim=0)
        #
        # elif self.char_vocab:
        #     chars_tensor = self.get_chars_tensor(words)
        #     words_tensor = torch.cat([chars_tensor, words_tensor.repeat(1)[:, None]], axis=1)

        return words_tensor, label_tensor

    # def get_chars_tensor(self, words):
    #     chars_tensor = []  # 20 (num of chars in each word)* 5 (num of words) = 100
    #     for word in words:
    #         chars_indices = self.char_vocab.get_chars_indexes_by_word(word)
    #         chars_tensor.append(chars_indices)
    #     chars_tensor = torch.tensor(chars_tensor).to(torch.int64)
    #     return chars_tensor


class SeqDataFile(Dataset):
    BASE_PATH = os.path.dirname(__file__)

    def __init__(self, examples_file, vocab: Vocab):
        self.data_path = os.path.join(self.BASE_PATH, examples_file)
        self.vocab = vocab
        self.data, self.labels = self.get_examples_and_labels()

    def get_examples_and_labels(self):
        examples = []
        labels = []
        with open(self.data_path, mode="r") as f:
            lines = f.readlines()
        for line in lines:
            example, label = line.strip().split("\t")
            examples.append(example)
            labels.append(label)

        return examples, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        word = self.data[index]
        label = self.labels[index]
        words_tensor = torch.tensor([self.vocab.get_word_index(w) for w in word]).to(torch.int64)
        label_tensor = torch.tensor([self.vocab.label2i[label]]).to(torch.int64)

        return words_tensor, label_tensor
