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
        guid: Unique id for the example.
        words: list. The words of the sequence.
        label: (Optional) str. The label of the middle word in the window
    """
    guid: str
    words: List[str]
    labels: Optional[List[str]]


class TokenDataFile(Dataset):
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

    def __init__(self, task: str, data_set, pre_processor: PreProcessor, vocab: Vocab,
                 sub_words: SubWords = None,
                 char_vocab: CharsVocab = None):
        self.task = task
        self.separator = " " if self.task == "pos" else "\t"
        self.data_path = os.path.join(self.BASE_PATH, task, data_set)
        self.pre_processor: PreProcessor = pre_processor
        self.vocab: Vocab = vocab
        self.sub_words = sub_words
        self.char_vocab = char_vocab
        self.data: List[InputExample] = self.read_examples_from_file()


class SeqDataFile(Dataset):
    BASE_PATH = os.path.dirname(__file__)

    def __init__(self, examples_file, vocab: Vocab):
        self.data_path = os.path.join(self.BASE_PATH, examples_file)
        self.vocab = vocab
        self.examples, self.labels = self.get_examples_and_labels()

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

    def __getitem__(self, index):
        word = self.examples[index]
        label = self.labels[index]
        words_tensor = torch.tensor([self.vocab.get_word_index(w) for w in word]).to(torch.int64)
        label_tensor = torch.tensor([self.vocab.label2i[label]]).to(torch.int64)

        return words_tensor, label_tensor
