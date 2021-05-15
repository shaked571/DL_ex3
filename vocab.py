import abc

class Vocab(abc.ABC):
    UNKNOWN_WORD = "UUUNKKK"

    def __init__(self):
        self.words, self.labels = self.get_unique()
        self.vocab_size = len(self.words)
        self.num_of_labels = len(self.labels)
        self.i2word = {i: w for i, w in enumerate(self.words)}
        self.word2i = {w: i for i, w in self.i2word.items()}
        self.i2label = {i: l for i, l in enumerate(self.labels)}
        self.label2i = {l: i for i, l in self.i2label.items()}

    def get_word_index(self, word):
        if word in self.word2i:
            return self.word2i[word]
        return self.word2i[self.UNKNOWN_WORD]

    @abc.abstractmethod
    def get_unique(self):
        pass


class TokenVocab(Vocab):
    def __init__(self, train_file: str, task: str):
        self.train_file = train_file
        self.task = task
        self.separator = " " if self.task == "pos" else "\t"
        super().__init__()


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
        words.update([self.UNKNOWN_WORD]) #TODO verify there is no need to add <s> and </s>
        labels.add('O')
        return words, labels


class CharVocab(Vocab):
    def __init__(self, train_file: str):
        self.train_file = train_file
        super().__init__()

    def get_unique(self):
        labels = {'0', '1'}
        nums = [str(i) for i in range(1, 10)]
        char = ['a', 'b', 'c', 'd']
        char += nums
        words = set(char)

        return words, labels
