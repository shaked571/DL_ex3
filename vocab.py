import abc


class Vocab(abc.ABC):
    UNKNOWN_TOKEN = "UUUNKKK"
    PAD_IDX = 0

    def __init__(self, task):
        self.task = task
        self.separator = " " if self.task == "pos" else "\t"
        self.tokens, self.labels = self.get_unique()
        self.tokens = list(self.tokens)
        self.tokens.insert(self.PAD_IDX, "PAD_DUMMY")
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
        words = set()
        labels = set()
        with open(self.train_file) as f:
            lines = f.readlines()
        for line in lines:
            if line == "" or line == "\n":
                continue
            word, label = line.strip().split(self.separator)
            words.add(word)
            labels.add(label)
        words.update([self.UNKNOWN_TOKEN])
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

#TODO################################################################
class SubWords(Vocab):
    def __init__(self, train_file: str, task: str):
        self.train_file = train_file
        super().__init__(task)

    def get_unique(self):
        chars = {self.UNKNOWN_TOKEN}
        with open(self.train_file) as f:
            lines = f.readlines()

        for line in lines:
            if line == "" or line == "\n":
                continue
            word, _ = line.strip().split(self.separator)
            chars.update([c for c in word])

        return chars, __

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
#TODO################################################################

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
