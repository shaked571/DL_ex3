import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from vocab import Vocab, SeqVocab, Binary, Num
from DataFiles import SeqDataFile
import argparse
from trainer import Trainer

class SeqLstm(nn.Module):

    def __init__(self, vocab: Vocab, embedding_dim=50, hidden_dim=100):
        super(SeqLstm, self).__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab.vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim)
        self.linear1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(self.hidden_dim, self.vocab.num_of_labels)


    def forward(self, x, x_lens):
        embeds = self.word_embeddings(x)
        x_packed = pack_padded_sequence(embeds, x_lens, batch_first=True, enforce_sorted=False)
        out, (ht, ct) = self.lstm(x_packed)
        out = self.linear1(ht[-1])
        out = self.tanh(out)
        out = self.linear2(out)
        return out


def main(train_file, test_file, optimizer='AdamW', batch_size=10, l_r=0.001,embedding_dim=20, hidden_dim=200, n_epochs=1,
         binaric=False,num=False ):
    task = "language"
    if binaric:
        vocab = Binary("binary")
    elif num:
        vocab = Num("binary")
    else:
        vocab = SeqVocab(task=task)
    model = SeqLstm(vocab,embedding_dim=embedding_dim, hidden_dim=hidden_dim)

    train_df = SeqDataFile(train_file, vocab)
    dev_df = SeqDataFile(test_file, vocab)

    trainer = Trainer(model=model,
                      train_data=train_df,
                      dev_data=dev_df,
                      lr=l_r,
                      train_batch_size=batch_size,
                      optimizer=optimizer,
                      vocab=vocab,
                      n_ep=n_epochs)
    trainer.train()
    trainer.evaluate_data_set(trainer.dev_data,"test on dev")
    trainer.evaluate_data_set(trainer.train_data, "test on train")

    test_prediction = trainer.test(dev_df)
    trainer.dump_test_file(test_prediction, dev_df.data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('train_file',help="train file name", type=str)
    parser.add_argument('dev_file', help="dev file name", type=str)
    parser.add_argument('-o', '--optimizer', type=str, required=False)
    parser.add_argument('-b', '--batch_size', type=int, required=False)
    parser.add_argument('-l', '--l_r', type=float, required=False)
    parser.add_argument('-h', '--hidden_dim', type=int, required=False)
    parser.add_argument('-d', '--embedding_dim', type=int, required=False)
    parser.add_argument('-e', '--epochs', type=int, required=False)
    parser.add_argument('--binaric', action="store_true", required=False)
    parser.add_argument('--num', action="store_true", required=False)

    args = parser.parse_args()

    main(train_file=args.train_file,
         test_file=args.dev_file,
         optimizer=args.optimizer,
         batch_size=args.batch_size,
         l_r=args.l_r,
         hidden_dim=args.hidden_dim,
         embedding_dim=args.embedding_dim,
         n_epochs=args.epochs,
         binaric=args.binaric,
         num=args.num)
