import argparse
from typing import List

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models import BiLSTMVanila
from trainer import Trainer
from vocab import TokenVocab
from DataFiles import TokenDataFile
torch.manual_seed(1)


def main(repr, train_file, dev_file, task, output_path, optimizer='AdamW', epochs=1, l_r=0.001, batch_size=10,
         embedding_dim=20, hidden_dim=200, dropout=0.2):
    vocab = TokenVocab(train_file, task)
    if repr == 'a':
        model = BiLSTMVanila(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab=vocab, dropout=dropout)
    else:
        raise ValueError(f"Not supporting repr: {repr} see help for details.")

    if dev_file is None:
        train_df = TokenDataFile(task, train_file,vocab, partial='train')
        dev_df = TokenDataFile(task, train_file, vocab,  partial='dev')
    else:
        train_df = TokenDataFile(task, train_file, vocab)
        dev_df = TokenDataFile(task, dev_file, vocab)

    trainer = Trainer(model=model,
                      train_data=train_df,
                      dev_data=dev_df,
                      lr=l_r,
                      train_batch_size=batch_size,
                      optimizer=optimizer,
                      vocab=vocab,
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
    parser.add_argument('-l', '--learning_rate', help='Number of epochs', default=0.001, type=float)
    parser.add_argument('-b', '--batch_size', help='Number of epochs', default=0.001, type=int)
    parser.add_argument('-do', '--drop_out', help='fropout value', default=0.2, type=float)
    args = parser.parse_args()
    main(
            repr=args.repr,
            train_file=args.trainFile,
            dev_file=args.dev_file,
            task=args.task,
            output_path=args.modelFile,
            optimizer=args.optimizer,
            epochs=args.epochs,
            l_r=args.learning_rate,
            batch_size=args.batch_size,
            dropout=args.drop_out
    )


