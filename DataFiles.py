import os
from dataclasses import dataclass
from typing import List, Optional
from torch.utils.data import DataLoader, Dataset

class Vocab:
    UNKNOWN_WORD = "UUUNKKK"
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    VOCAB_PATH = os.path.join(BASE_PATH, 'vocab.txt')

    def __init__(self, task: str, word2vec):
        self.task = task
        self.separator = " " if self.task == "pos" else "\t"
        self.word2vec = word2vec
        self.train_path = os.path.join(self.BASE_PATH, self.task, 'train')
        if self.word2vec:
            _, self.labels = self.get_unique(self.train_path)
            self.words = self.get_word2vec_words()
        else:
            self.words, self.labels = self.get_unique(self.train_path)

        self.vocab_size = len(self.words)
        self.num_of_labels = len(self.labels)
        self.i2word = {i: w for i, w in enumerate(self.words)}
        self.word2i = {w: i for i, w in self.i2word.items()}
        self.i2label = {i: l for i, l in enumerate(self.labels)}
        self.label2i = {l: i for i, l in self.i2label.items()}

    def get_word_index(self, word):
        if self.word2vec:
            word = word.lower()

        if word in self.word2i:
            return self.word2i[word]

        return self.word2i[self.UNKNOWN_WORD]

    def get_word2vec_words(self):
        vocab = []
        with open(self.VOCAB_PATH) as f:
            lines = f.readlines()
        for line in lines:
            word = line.strip()
            vocab.append(word)
        return vocab

    def get_unique(self, path):
        words = set()
        labels = set()
        with open(path) as f:
            lines = f.readlines()
        for line in lines:
            if line == "" or line == "\n":
                continue
            word, label = line.strip().split(self.separator)
            words.add(word)
            labels.add(label)
        words.update(["</s>", "<s>", self.UNKNOWN_WORD])
        labels.add('O')

        return words, labels

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
