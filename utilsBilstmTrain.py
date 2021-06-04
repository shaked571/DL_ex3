import os
import torch
from torch.utils.data import TensorDataset

abspath = os.path.dirname(os.path.abspath(__file__))


def read_data(fname, mode=None):
    f = open(abspath + '/' + fname, "r")
    lines = f.read().split('\n')

    data = []
    sequence = []

    for line in lines:
        if line == '':
            if len(sequence) > 0:
                data.append(sequence)
                sequence = []
        else:
            if mode == 'test':
                word, label = line, None
            else:
                word, label = line.split()
            sequence.append((word, label))

    return data


class DatasetTagger:
    def __init__(self, args):
        self.train_data = read_data('data/' + args.task_name + '/train')
        self.dev_data = read_data('data/' + args.task_name + '/dev')
        self.prefix_vocab = None
        self.suffix_vocab = None

        # set.union({'UUUNKKK', '<s>', '</s>'}

        self.vocab = ['<pad>'] + list(set.union({'UUUNKKK'}, set(word for word, label in [item for sequence in self.train_data for item in sequence])))
        self.labels = list(set(label for word, label in [item for sequence in self.train_data for item in sequence]))

        if args.adding_sub_word_unit:
            train_vocab = set(self.vocab) - {'<pad>', '<s>', '</s>'}
            self.prefix_vocab = ['<pad>'] + list(set.union({'UUUNKKK'}, set([word[:3] for word in train_vocab])))
            self.suffix_vocab = ['<pad>'] + list(set.union({'UUUNKKK'}, set([word[-3:] for word in train_vocab])))
            self.P2I = {w: i for i, w in enumerate(self.prefix_vocab)}
            self.S2I = {w: i for i, w in enumerate(self.suffix_vocab)}

        self.W2I = {w: i for i, w in enumerate(self.vocab)}
        self.L2I = {l: i for i, l in enumerate(self.labels)}
        self.I2L = {i: l for i, l in enumerate(self.labels)}

    def convert_data_to_examples(self, args, mode):
        all_input_ids = []
        all_prefix_ids = []
        all_suffix_ids = []
        all_seq_len = []
        all_label_ids = []

        if mode == 'train':
            data = self.train_data
        else:  # mode == 'dev'
            data = self.dev_data

        max_len_seq = len(max(data, key=len))  # +2 for '<s>', '</s>'

        print(max_len_seq)

        for example in data:
            input_ids = []
            prefix_ids = []
            suffix_ids = []
            label_ids = []

            # input_ids.append(self.W2I['<s>'])
            # label_ids.append(-1)

            for word, label in example:
                word_id = self.W2I['UUUNKKK']
                if word in self.W2I:
                    word_id = self.W2I[word]
                label_id = self.L2I[label]
                input_ids.append(word_id)
                label_ids.append(label_id)

                if args.adding_sub_word_unit:
                    prefix_id = self.P2I['UUUNKKK']
                    suffix_id = self.S2I['UUUNKKK']
                    if word[:3] in self.P2I:
                        prefix_id = self.P2I[word[:3]]
                    if word[-3:] in self.S2I:
                        suffix_id = self.S2I[word[-3:]]
                    prefix_ids.append(prefix_id)
                    suffix_ids.append(suffix_id)

            # input_ids.append(self.W2I['</s>'])
            # label_ids.append(-1)

            seq_len = len(input_ids)

            input_ids += [self.W2I['<pad>']] * (max_len_seq - len(input_ids))
            if args.adding_sub_word_unit:
                prefix_ids += [self.P2I['<pad>']] * (max_len_seq - len(prefix_ids))
                suffix_ids += [self.S2I['<pad>']] * (max_len_seq - len(suffix_ids))
            label_ids += [-1] * (max_len_seq - len(label_ids))

            all_input_ids.append(input_ids)
            all_seq_len.append(seq_len)
            all_prefix_ids.append(prefix_ids)
            all_suffix_ids.append(suffix_ids)
            all_label_ids.append(label_ids)

        dataset = TensorDataset(torch.tensor(all_input_ids, dtype=torch.long),
                                torch.tensor(all_seq_len, dtype=torch.long),
                                torch.tensor(all_prefix_ids, dtype=torch.long),
                                torch.tensor(all_suffix_ids, dtype=torch.long),
                                torch.tensor(all_label_ids, dtype=torch.long))

        return dataset
