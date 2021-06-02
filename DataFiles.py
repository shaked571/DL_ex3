import os
from dataclasses import dataclass
from typing import List, Optional
from torch.utils.data import DataLoader, Dataset
import torch
from vocab import Vocab, CharsVocab, SubWords


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

    def __init__(self,
                 task: str,
                 data_fname: str,
                 mission: str,
                 vocab: Vocab = None,
                 partial=None,
                 sub_words: SubWords = None,
                 chars_vocab: CharsVocab = None
                 ):
        self.data_path = data_fname
        self.separator = " " if task == "pos" else "\t"
        self.mission = mission
        self.vocab = vocab
        self.sub_words = sub_words
        self.char_vocab = chars_vocab
        self.data = self.get_examples_and_labels()
        if partial == 'train':
            self.data = self.data[:int(len(self.data) * 0.9)]
        elif partial == 'dev':
            self.data = self.data[int(len(self.data) * 0.9):]

    def read_sents(self, lines):
        sentences = []
        sent = []
        for line in lines:
            if line == "" or line == "\n":
                if sent:
                    sentences.append(sent)
                    sent = []
            else:
                sent.append(tuple(line.strip().split(self.separator)))
        return sentences

    def get_examples_and_labels(self):
        examples = []
        with open(self.data_path, mode="r") as f:
            lines = f.readlines()

        curr_words = []
        curr_labels = []
        guid_index = 0
        for line in lines:
            if line == "" or line == "\n":
                if curr_words:
                    guid_index += 1
                    example = InputExample(guid=f"{self.data_path}-{guid_index}", words=curr_words, labels=curr_labels)
                    examples.append(example)
                    curr_words = []
                    curr_labels = []
            else:
                if 'test' in self.data_path:
                    word = line.strip()
                    label = 'O'
                else:
                    word, label = tuple(line.strip().split(self.separator))
                curr_words.append(word)
                curr_labels.append(label)

        return examples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        words = self.data[index].words
        labels = self.data[index].labels

        labels = ['O'] + labels + ['O']
        words = ['<s>'] + words + ['<\s>']
        words_tensor = torch.tensor([self.vocab.get_word_index(word) for word in words]).to(torch.int64)
        label_tensor = torch.tensor([self.vocab.label2i[label] for label in labels]).to(torch.int64)

        return words_tensor, label_tensor

    def get_chars_tensor(self, words):
        chars_tensor = []
        max_len = max([len(w) for w in words])
        for word in words:
            chars_indices = self.char_vocab.get_chars_indexes_by_word(word)
            chars_tensor.append(chars_indices)
        for c_w in chars_tensor:
            if len(c_w) < max_len:
                c_w += [0] * (max_len - len(c_w))
        chars_tensor = torch.tensor(chars_tensor).to(torch.int64)
        return chars_tensor


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
