import abc
from collections import Counter


class Vocab(abc.ABC):
    UNKNOWN_TOKEN = "UUUNKKK"
    PAD_DUMMY = "PAD_DUMMY"
    PAD_IDX = 0

    def __init__(self, task):
        self.task = task
        self.separator = " " if self.task == "pos" else "\t"
        self.tokens, self.labels = self.get_unique()
        self.tokens = list(self.tokens)
        self.tokens.insert(self.PAD_IDX, self.PAD_DUMMY)
        self.vocab_size = len(self.tokens)
        self.num_of_labels = len(self.labels)
        self.i2token = {i: w for i, w in enumerate(self.tokens)}
        self.token2i = {w: i for i, w in self.i2token.items()}
        self.i2label = {i: l for i, l in enumerate(self.labels)}
        self.label2i = {l: i for i, l in self.i2label.items()}

    def get_word_index(self, word):
        if word in self.token2i:
            return self.token2i[word]
        return self.token2i[self.UNKNOWN_TOKEN]

    @abc.abstractmethod
    def get_unique(self):
        pass


class TokenVocab(Vocab):

    def __init__(self, train_file: str, task: str):
        self.train_file = train_file
        super().__init__(task)

    def get_unique(self):
        words = []

        labels = set()
        with open(self.train_file) as f:
            lines = f.readlines()
        for line in lines:
            if line == "" or line == "\n":
                continue
            word, label = line.strip().split(self.separator)
            words.append(word)
            labels.add(label)
        not_single = [k for k, v in Counter(words).items() if v != 1 ]
        not_single.append(self.UNKNOWN_TOKEN)
        words = not_single
        labels.add('O')
        return words, labels


class CharsVocab(Vocab):
    def __init__(self, train_file: str, task: str):
        self.train_file = train_file
        super().__init__(task)

    def get_unique(self):
        chars = {self.UNKNOWN_TOKEN}
        labels = set()

        with open(self.train_file) as f:
            lines = f.readlines()

        for line in lines:
            if line == "" or line == "\n":
                continue
            word, label = line.strip().split(self.separator)
            chars.update([c for c in word])
            labels.add(label)

        labels.add('O')
        return chars, labels

    def get_chars_indexes_by_word(self, word):
        word_chars = [c for c in word]
        indexes = []
        # add chars indexes
        for c in word_chars:
            if c in self.token2i:
                indexes.append(self.token2i[c])
            else:
                indexes.append(self.token2i[self.UNKNOWN_TOKEN])
        return indexes


class SubWords:
    SUB_WORD_SIZE = 3
    SHORT_SUB_WORD = "SHORT_WORD"
    UNKNOWN_SUB_WORD = "UNKNOW_SUB_WORD"
    PAD_DUMMY = "PAD_DUMMY"
    PAD_IDX = 0

    def __init__(self, train_file: str, task: str):
        self.train_file = train_file
        self.separator = " " if task == "pos" else "\t"
        self.suffix, self.prefix = self.get_prefix_and_suffix()

        self.suffix.insert(self.PAD_IDX, self.PAD_DUMMY)
        self.prefix.insert(self.PAD_IDX, self.PAD_DUMMY)

        self.suffix_num = len(self.suffix)
        self.prefix_num = len(self.prefix)

        self.suffix2i = {s: i for i, s in enumerate(self.suffix)}
        self.i2suffix = {i: s for i, s in enumerate(self.suffix)}
        self.prefix2i = {p: i for i, p in enumerate(self.prefix)}
        self.i2prefix = {i: p for i, p in enumerate(self.prefix)}

    def get_unique(self):
        chars = {self.UNKNOWN_SUB_WORD}
        with open(self.train_file) as f:
            lines = f.readlines()

        for line in lines:
            if line == "" or line == "\n":
                continue
            word, _ = line.strip().split(self.separator)
            chars.update([c for c in word])

        return chars, _

    def get_sub_words_indexes_by_word(self, word):
        if word == self.PAD_DUMMY:
            return 0, 0

        if len(word) < self.SUB_WORD_SIZE:
            return self.prefix2i[self.SHORT_SUB_WORD], self.suffix2i[self.SHORT_SUB_WORD]

        prefix, suffix = self.get_sub_words_by_word(word)
        if prefix not in self.prefix2i:
            prefix = self.UNKNOWN_SUB_WORD
        if suffix not in self.suffix2i:
            suffix = self.UNKNOWN_SUB_WORD

        return self.prefix2i[prefix], self.suffix2i[suffix]

    def get_sub_words_by_word(self, word):
        suffix = word[len(word) - self.SUB_WORD_SIZE:]
        prefix = word[:self.SUB_WORD_SIZE]
        return prefix, suffix

    def get_prefix_and_suffix(self):
        suffixes = {self.SHORT_SUB_WORD, self.UNKNOWN_SUB_WORD}
        prefixes = {self.SHORT_SUB_WORD, self.UNKNOWN_SUB_WORD}

        with open(self.train_file) as f:
            lines = f.readlines()

        for line in lines:
            if line == "" or line == "\n":
                continue
            word, _ = line.strip().split(self.separator)

            if len(word) < self.SUB_WORD_SIZE:
                continue

            prefix, suffix = self.get_sub_words_by_word(word)
            suffixes.add(suffix)
            prefixes.add(prefix)

        return list(prefixes), list(suffixes)


class SeqVocab(Vocab):
    def __init__(self, task: str):
        super().__init__(task)

    def get_unique(self):
        labels = {'0', '1'}
        nums = [str(i) for i in range(1, 10)]
        char = ['a', 'b', 'c', 'd']
        char += nums
        words = set(char)

        return words, labels


class Binary(Vocab):
    def __init__(self, task: str):
        super().__init__(task)

    def get_unique(self):
        labels = {'0', '1'}
        words = ['0', '1', 'b']
        return words, labels


class Num(Vocab):
    def __init__(self, task: str):
        super().__init__(task)

    def get_unique(self):
        labels = {'0', '1'}
        words = [str(i) for i in range(10)]
        return words, labels
