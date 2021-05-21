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
from vocab import Vocab

torch.manual_seed(1)

repr
train_file
output_path
optimizer
epochs
learning_rate
batch_size

def main(repr, train_file,dev_file, task, output_path ,optimizer='AdamW', epochs=1, l_r=0.001,batch_size=10, embedding_dim=20, hidden_dim=200,
         dropout=0.2):
    vocab = Vocab(task)
    if repr == 'a':
        BiLSTMVanila(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab=vocab, dropot=dropout)
    model = SeqLstm(vocab, embedding_dim=embedding_dim, hidden_dim=hidden_dim)

    train_df = SeqDataFile(train_file, vocab)
    dev_df = SeqDataFile(dev_file, vocab)
    test_df = SeqDataFile(test_file, vocab)
    test_data = DataLoader(test_df, batch_size=batch_size, collate_fn=pad_collate)

    trainer = Trainer(model=model,
                      train_data=train_df,
                      dev_data=dev_df,
                      lr=l_r,
                      train_batch_size=batch_size,
                      optimizer=optimizer,
                      vocab=vocab,
                      n_ep=n_epochs)
    trainer.train()
    trainer.evaluate_data_set(trainer.train_data, "test on train")
    trainer.evaluate_data_set(trainer.dev_data,"test on dev")
    trainer.evaluate_data_set(test_data, "test on test")

    test_prediction = trainer.test(test_df)
    trainer.dump_test_file(test_prediction, test_df.data_path)



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
    parser.add_argument('-b', '--batch_size', help='Number of epochs', default=0.001, type=float)
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
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            dropout=args.drop_out
    )


