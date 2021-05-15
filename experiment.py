import torch.nn as nn
import torch.nn.functional as F
from vocab import Vocab, SeqVocab
from DataFiles import SeqDataFile
import argparse
from trainer import Trainer


class SeqLstm(nn.Module):

    def __init__(self, vocab: Vocab, embedding_dim=30, hidden_dim=100):
        super(SeqLstm, self).__init__()
        self.vocab = vocab

        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab.vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        # self.hidden = nn.Linear(self.hidden_dim, self.hidden_dim)

        # mlp layers
        self.linear1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(self.hidden_dim, self.vocab.num_of_labels)

    def forward(self, x):
        embeds = self.word_embeddings(x)
        out, _ = self.lstm(embeds.view(len(x), 1, -1))
        # out = self.hidden(out.view(len(x), -1))
        # out = F.log_softmax(out, dim=1)

        # out = out.view(len(x), -1)
        out = out[-1][-1]
        out = self.linear1(out)
        out = self.tanh(out)
        out = self.linear2(out)
        return out


def main(train_file, test_file, optimizer, batch_size, l_r, hidden_dim):
    task = "language"
    vocab = SeqVocab(task=task)
    model = SeqLstm(vocab)

    train_df = SeqDataFile(train_file, vocab)
    dev_df = SeqDataFile(test_file, vocab)

    trainer = Trainer(model=model,
                      train_data=train_df,
                      dev_data=dev_df,
                      train_batch_size=None,
                      vocab=vocab,
                      n_ep=5)
    trainer.train()
    test_prediction = trainer.test(dev_df)
    trainer.dump_test_file(test_prediction, dev_df.data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-t', '--train_file', help="train file name", type=str, required=True)
    parser.add_argument('-d', '--dev_file', help="dev file name", type=str, required=True)
    parser.add_argument('--optimizer', type=str, required=False)
    parser.add_argument('--batch_size', type=int, required=False)
    parser.add_argument('--l_r', type=float, required=False)
    parser.add_argument('--hidden_dim', type=int, required=False)

    args = parser.parse_args()

    main(args.train_file, args.dev_file, args.optimizer, args.batch_size, args.l_r, args.hidden_dim)
