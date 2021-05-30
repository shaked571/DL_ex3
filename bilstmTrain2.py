from utilsBilstmTrain import DatasetTagger

import os
import random
import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm, trange

abspath = os.path.dirname(os.path.abspath(__file__))
debug = False


def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", default='ner', type=str, required=False,
                        help="The name of the task to train, selected from: [pos, ner]")
    parser.add_argument("--word2vec_embedding",  # action='store_true',
                        type=lambda x: (str(x).lower() == 'true'),
                        default=False,
                        help="Whether using word2vec embeddings.")
    parser.add_argument("--adding_sub_word_unit",  # action='store_true',
                        type=lambda x: (str(x).lower() == 'true'),
                        default=False,
                        help="Whether adding sub word unit.")
    parser.add_argument("--adding_character_level_lstm",  # action='store_true',
                        type=lambda x: (str(x).lower() == 'true'),
                        default=False,
                        help="Whether adding character level LSTM.")
    parser.add_argument("--learning_rate", default=0.01, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", default=100, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=1024, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    return args


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)


class BiLSTMTagger(nn.Module):
    def __init__(self,
                 args,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 tag_size,
                 prefix_vocab_size=None,
                 suffix_vocab_size=None):
        super(BiLSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if args.adding_sub_word_unit:
            self.prefix_embeddings = nn.Embedding(prefix_vocab_size, embedding_dim, padding_idx=0)
            self.suffix_embeddings = nn.Embedding(suffix_vocab_size, embedding_dim, padding_idx=0)
            nn.init.xavier_uniform_(self.prefix_embeddings.weight)
            nn.init.xavier_uniform_(self.suffix_embeddings.weight)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        # self.dropout = nn.Dropout(0.3)
        self.linear1 = nn.Linear(2 * hidden_dim, tag_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, input_ids, seq_lengths, prefix_ids, suffix_ids):
        embeddings = self.embeddings(input_ids)

        if prefix_ids.shape[1] > 0:
            embeddings += self.prefix_embeddings(prefix_ids)

        if suffix_ids.shape[1] > 0:
            embeddings += self.suffix_embeddings(suffix_ids)

        # h0 = torch.zeros(4, input_ids.size(0), self.hidden_dim).to('cuda')
        # c0 = torch.zeros(4, input_ids.size(0), self.hidden_dim).to('cuda')

        packed_input = pack_padded_sequence(embeddings, seq_lengths.cpu().numpy(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed_input)  # (h0, c0)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        linear1_out = self.linear1(lstm_out)
        log_probs = linear1_out.reshape((linear1_out.shape[0] * linear1_out.shape[1], linear1_out.shape[2]))
       # print(log_probs.shape)
        return log_probs


def accuracy_on_dataset(args, dataset, model):
    good = bad = 0.0
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    total_loss = 0

    #loss_function = nn.CrossEntropyLoss(ignore_index=-1)

    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=True):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        curr_predictions = model(batch[0], batch[1], batch[2], batch[3]).detach().cpu().numpy()

        # with torch.no_grad():
        #     log_probs = model(batch[0], batch[1], batch[2], batch[3])
        #     loss = loss_function(log_probs, batch[4])
        #     total_loss += loss.item()

        labels_id = torch.flatten(batch[4][:, torch.LongTensor(list(range(0, max(batch[1]))))]).detach().cpu().numpy()

        for prediction, label in (list(zip(curr_predictions, labels_id))):
            if label == -1:
                continue
            if label == np.argmax(prediction):
                if args.task_name == 'ner' and label == args.majorClassIndex:
                    continue
                good += 1
            else:
                bad += 1

    # dev_loss = total_loss / len(dataset)

    return good / (good + bad), -1


def train(args, train_dataset, dev_dataset, model):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=True)

    #loss_function = nn.NLLLoss(ignore_index=-1, reduction='mean')
    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters())

    I = 0
    dev_loss_arr = []
    dev_accuracy_arr = []

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)

        total_loss = 0

        model.train()

        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)

            # zero out the gradients from the old instance
            model.zero_grad()

            # run the forward pass, getting log probabilities over tag
            log_probs = model(batch[0], batch[1], batch[2], batch[3])

            # compute loss function
            labels_id = torch.flatten(batch[4][:, torch.LongTensor(list(range(0, max(batch[1]))))])
            loss = loss_function(log_probs, labels_id) # reduction='sum'

            # do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            # get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()

        I = I + 1

        train_loss = total_loss / len(train_dataset)
        train_accuracy, _train_loss = accuracy_on_dataset(args, train_dataset, model)
        dev_accuracy, dev_loss1 = accuracy_on_dataset(args, dev_dataset, model)

        print(I, train_loss, train_accuracy, dev_accuracy)

        # if debug:
        #     dev_loss_arr.append(dev_loss)
        #     dev_accuracy_arr.append(dev_accuracy)
        #
        #     epochs = range(1, I + 1)
        #     plt.plot(epochs, dev_loss_arr, 'g', label='Dev loss')
        #     plt.xlabel('Epochs')
        #     plt.ylabel('Loss')
        #     plt.legend()
        #     plt.savefig('loss_' + str(I) + '.png')
        #     plt.clf()
        #
        #     plt.plot(epochs, dev_accuracy_arr, 'b', label='Dev accuracy')
        #     plt.xlabel('Epochs')
        #     plt.ylabel('Accuracy')
        #     plt.legend()
        #     plt.savefig('accuracy_' + str(I) + '.png')
        #     plt.clf()


def main():
    embedding_dim = 50
    hidden_dim = 50

    args = init_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args)

    data_set_tagger = DatasetTagger(args)

    if args.task_name == 'ner':
        args.majorClassIndex = data_set_tagger.L2I['O']

    model = BiLSTMTagger(args=args,
                         vocab_size=len(data_set_tagger.vocab),
                         embedding_dim=embedding_dim,
                         hidden_dim=hidden_dim,
                         tag_size=len(data_set_tagger.labels),
                         prefix_vocab_size=None if data_set_tagger.prefix_vocab is None else len(data_set_tagger.prefix_vocab),
                         suffix_vocab_size=None if data_set_tagger.prefix_vocab is None else len(data_set_tagger.suffix_vocab))

    model.to(args.device)

    train(args=args,
          train_dataset=data_set_tagger.convert_data_to_examples(args, mode='train'),
          dev_dataset=data_set_tagger.convert_data_to_examples(args, mode='dev'),
          model=model)


if __name__ == '__main__':
    main()
