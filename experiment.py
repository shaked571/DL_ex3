import torch.nn as nn
import torch.nn.functional as F
from vocab import Vocab, SeqVocab
import torch
from DataFiles import SeqDataFile
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse


class SeqLstm(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab: Vocab):
        super(SeqLstm, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab.vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden = nn.Linear(hidden_dim, vocab.num_of_labels)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


class Trainer:

    def __init__(self, model: nn.Module, train_data: SeqDataFile, dev_data: SeqDataFile, vocab: Vocab, n_ep=1,
                 optimizer='AdamW', train_batch_size=8, steps_to_eval=30000, lr=0.01, filter_num=30, window_size=3, part=None):
        self.part = part
        self.model = model
        self.dev_batch_size = 128
        self.train_data = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
        self.dev_data = DataLoader(dev_data, batch_size=self.dev_batch_size, )
        self.vocab = vocab
        if optimizer == "SGD":
            self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
        elif optimizer == "AdamW":
            self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        elif optimizer == "Adam":
            self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        else:
            raise ValueError("optimizer supports SGD, Adam, AdamW")
        self.steps_to_eval = steps_to_eval
        self.n_epochs = n_ep
        self.loss_func = nn.CrossEntropyLoss()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        # self.model_args = {"part": self.part, "task": self.vocab.task, "lr": lr, "epoch": self.n_epochs,
        #                    "batch_size": train_batch_size, "filter_num": filter_num, "window_size": window_size,
        #                    "steps_to_eval": self.steps_to_eval, "optim": optimizer, "hidden_dim": self.model.hidden_dim}
        # self.writer = SummaryWriter(log_dir=f"tensor_board/{self.suffix_run()}")
        #
        # self.saved_model_path = f"{self.suffix_run()}.bin"

        self.best_model = None
        self.best_score = 0

    def train(self):
        for epoch in range(self.n_epochs):
            ###################
            # train the model #
            ###################
            print(f"start epoch: {epoch + 1}")
            train_loss = 0.0
            step_loss = 0
            self.model.train()  # prep model for training
            for step, (data, target) in tqdm(enumerate(self.train_data), total=len(self.train_data)):
                data = data.to(self.device)
                target = target.to(self.device)
                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)  # Eemnded Data Tensor size (1,5)
                # calculate the loss
                loss = self.loss_func(output, target.view(-1))
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                # update running training loss
                train_loss += loss.item() * data.size(0)
                step_loss += loss.item() * data.size(0)
                if step % self.steps_to_eval == 0:
                    print(f"in step: {step} train loss: {step_loss}")
                    # self.writer.add_scalar('Loss/train_step', step_loss, step * (epoch + 1))
                    step_loss = 0.0
                    self.evaluate_model(step * (epoch + 1), "step")
            print(f"in epoch: {epoch + 1} train loss: {train_loss}")
            # self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.evaluate_model(epoch, "epoch")

    def evaluate_model(self, step, stage):
        with torch.no_grad():
            self.model.eval()
            loss = 0

            prediction = []
            all_target = []
            for eval_step, (data, target) in tqdm(enumerate(self.dev_data), total=len(self.dev_data),
                                                  desc=f"dev step {step} loop"):
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)

                loss = self.loss_func(output, target.view(-1))
                loss += loss.item() * data.size(0)
                _, predicted = torch.max(output, 1)
                prediction += predicted.tolist()
                all_target += target.view(-1).tolist()
            accuracy = self.accuracy_token_tag(prediction, all_target)
            print(f'Accuracy/dev_{stage}: {accuracy}')
            self.writer.add_scalar(f'Accuracy/dev_{stage}', accuracy, step)
            self.writer.add_scalar(f'Loss/dev_{stage}', loss, step)
            if accuracy > self.best_score:
                self.best_score = accuracy
                torch.save(self.model.state_dict(), self.saved_model_path)

        self.model.train()

def main(task, part, optimizer, batch_size, l_r, hidden_dim):
    embedding_dim = 50
    char_embedding_dim = 30
    filter_num = 30
    window_size = 3
    word2vec = False
    if part == 3:
        word2vec = True

    vocab = (task, word2vec)

    sub_words = None
    char_vocab = None
    if part == 4:
        sub_words = SubWords(task)
        model = MLPSubWords(embedding_dim, hidden_dim, vocab, sub_words)
    elif part == 5:
        char_vocab = CharsVocab(task)
        model = CnnMLPSubWords(embedding_dim, hidden_dim, vocab, char_embedding_dim, filter_num, window_size, char_vocab)
    else:
        model = MLP(embedding_dim, hidden_dim, vocab)

    train_df = DataFile(task, 'train', title_process, vocab, sub_words, char_vocab)
    dev_df = DataFile(task, 'dev', title_process, vocab, sub_words, char_vocab)

    trainer = Trainer(model=model,
                      train_data=train_df,
                      dev_data=dev_df,
                      vocab=vocab,
                      n_ep=7,
                      optimizer=optimizer,
                      train_batch_size=batch_size,
                      lr=l_r,
                      filter_num=filter_num,
                      window_size=window_size,
                      part=part)
    trainer.train()
    test_prediction = trainer.test(test_df)
    trainer.dump_test_file(test_prediction, test_df.data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--optimizer', type=str, required=False)
    parser.add_argument('--batch_size', type=int, required=False)
    parser.add_argument('--l_r', type=float, required=False)
    parser.add_argument('--hidden_dim', type=int, required=True)

    args = parser.parse_args()

    main(args.task, args.part, args.optimizer, args.batch_size, args.l_r, args.hidden_dim)
