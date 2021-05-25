import argparse
from typing import List

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models import BiLSTMVanila, BiLSTMChar
from trainer import Trainer
from vocab import TokenVocab, CharsVocab
from DataFiles import TokenDataFile
torch.manual_seed(1)


def main(mission, train_file_name, dev_file_name, task, output_path, optimizer='AdamW', epochs=1, l_r=0.001, batch_size=10,
         embedding_dim=20, hidden_dim=100, dropout=0.2, sent_len=128, lstm_hidden_dim=50):

    chars_vocab = None
    sub_words = None

    vocab = TokenVocab(train_file_name, task)
    if mission == 'a':
        model = BiLSTMVanila(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab=vocab, dropout=dropout,
                             sent_len=sent_len)
    elif mission == 'b':
        chars_vocab = CharsVocab(train_file_name, task)
        model = BiLSTMChar(embedding_dim=embedding_dim, hidden_dim=hidden_dim,  lstm_hidden_dim=lstm_hidden_dim, vocab=vocab, chars_vocab=chars_vocab,
                           dropout=dropout, sent_len=sent_len)
    else:
        raise ValueError(f"Not supporting repr: {mission} see help for details.")

    if dev_file_name is None:
        train_df = TokenDataFile(task=task, data_fname=train_file_name, mission=mission, vocab=vocab, partial='train',
                                 sub_words=sub_words, chars_vocab=chars_vocab)
        dev_df = TokenDataFile(task=task, data_fname=train_file_name, mission= mission, vocab=vocab, partial='dev',
                               sub_words=sub_words, chars_vocab=chars_vocab)
    else:
        train_df = TokenDataFile(task=task, data_fname=train_file_name, mission=mission, vocab=vocab,
                                 sub_words=sub_words, chars_vocab=chars_vocab)
        dev_df = TokenDataFile(task=task, data_fname=dev_file_name, mission=mission, vocab=vocab,
                               sub_words=sub_words, chars_vocab=chars_vocab)

    trainer = Trainer(model=model,
                      train_data=train_df,
                      dev_data=dev_df,
                      lr=l_r,
                      train_batch_size=batch_size,
                      optimizer=optimizer,
                      vocab=vocab,
                      char_vocab = chars_vocab,
                      output_path=output_path,
                      n_ep=epochs)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aligner model')
    parser.add_argument('repr', help="repr need to be {a,b,c,d} when\n"
                                     "a: an embedding vector.\n"
                                     "b: a character-level LSTM.\n"
                                     "c: the embeddings+subword representation used in assignment 2.\n"
                                     "d: a concatenation of (a) and (b) followed by a linear layer.")
    parser.add_argument('trainFile', help="The input file to train on.")
    parser.add_argument('modelFile', help="the file to save the model.")
    parser.add_argument('-dev', '--dev_file', help="evaluation (development) data set.")
    parser.add_argument('-t', '--task', help="{pos, ner}")
    parser.add_argument('-o', '--optimizer', type=str, default='AdamW')
    parser.add_argument('-e', '--epochs', help='Number of epochs', default=5, type=int)
    parser.add_argument('-s', '--sent_len', help='Max length of sentence', default=128, type=int)
    parser.add_argument('-l', '--learning_rate', help='Number of epochs', default=0.001, type=float)
    parser.add_argument('-b', '--batch_size', help='Number of epochs', default=0.001, type=int)
    parser.add_argument('-do', '--drop_out', help='fropout value', default=0.2, type=float)
    parser.add_argument('-lhd', '--lstm_hidden_dim', help='lstm hidden dimension value', default=50, type=int)
    parser.add_argument('-hd', '--hidden_dim', help='main hidden dimension value', default=50, type=int)
    args = parser.parse_args()
    main(mission=args.repr,
         train_file_name=args.trainFile,
         dev_file_name=args.dev_file,
         task=args.task,
         output_path=args.modelFile,
         optimizer=args.optimizer,
         epochs=args.epochs,
         l_r=args.learning_rate,
         batch_size=args.batch_size,
         dropout=args.drop_out,
         lstm_hidden_dim=args.lstm_hidden_dim,
         hidden_dim=args.hidden_dim,
         sent_len=args.sent_len)


