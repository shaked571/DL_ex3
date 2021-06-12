import torch
from torch.utils.data import DataLoader

from models import SeqLstm
from vocab import SeqVocab, Binary, Num
from DataFiles import SeqDataFile
import argparse
from trainer import Trainer
from trainer import pad_collate

torch.manual_seed(1)


def main(train_file, dev_file, test_file, optimizer='AdamW', batch_size=10, l_r=0.001,embedding_dim=20, hidden_dim=200, n_epochs=1,
         binaric=False, num=False):
    task = "language"
    if binaric:
        vocab = Binary("binary")
    elif num:
        vocab = Num("binary")
    else:
        vocab = SeqVocab(task=task)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('train_file',help="train file name", type=str)
    parser.add_argument('dev_file', help="dev file name", type=str)
    parser.add_argument('test_file', help="test file name", type=str)
    parser.add_argument('-o', '--optimizer', type=str,default='AdamW', required=False)
    parser.add_argument('-b', '--batch_size', type=int, default=10, required=False)
    parser.add_argument('-l', '--l_r', type=float,default=0.001, required=False)
    parser.add_argument('-h', '--hidden_dim', type=int,default=200, required=False)
    parser.add_argument('-d', '--embedding_dim', default=20, type=int, required=False)
    parser.add_argument('-e', '--epochs', type=int, default=10, required=False)
    parser.add_argument('--binaric', action="store_true", required=False)
    parser.add_argument('--num', action="store_true", required=False)

    args = parser.parse_args()

    main(train_file=args.train_file,
         dev_file=args.dev_file,
         test_file=args.test_file,
         optimizer=args.optimizer,
         batch_size=args.batch_size,
         l_r=args.l_r,
         hidden_dim=args.hidden_dim,
         embedding_dim=args.embedding_dim,
         n_epochs=args.epochs,
         binaric=args.binaric,
         num=args.num)
