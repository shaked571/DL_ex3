import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import os
from math import sqrt







class Trainer:
    def __init__(self, model: nn.Module, train_data: Dataset, dev_data: Dataset, vocab: Vocab, n_ep=1,
                 optimizer='AdamW', train_batch_size=8, steps_to_eval=500, lr=0.01, part=None):
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
        self.loss_func = nn.CrossEntropyLoss(ignore_index=F2I[PAD]) #PU HERE THE PADDINF
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model_args = {"part": self.part, "task": self.vocab.task, "lr": lr, "epoch": self.n_epochs,
                           "batch_size": train_batch_size, "filter_num": filter_num, "window_size": window_size,
                           "steps_to_eval": self.steps_to_eval, "optim": optimizer, "hidden_dim": self.model.hidden_dim}
        self.writer = SummaryWriter(log_dir=f"tensor_board/{self.suffix_run()}")

        self.saved_model_path = f"{self.suffix_run()}.bin"

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
                    self.writer.add_scalar('Loss/train_step', step_loss, step * (epoch + 1))
                    step_loss = 0.0
                    self.evaluate_model(step * (epoch + 1), "step")
            print(f"in epoch: {epoch + 1} train loss: {train_loss}")
            self.writer.add_scalar('Loss/train', train_loss, epoch)
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

    def suffix_run(self):
        res = ""
        for k, v in self.model_args.items():
            res += f"{k}_{v}_"
        res = res.strip("_")
        return res

    def test(self, test_df):
        test = DataLoader(test_df, batch_size=self.dev_batch_size, )
        self.model.load_state_dict(torch.load(self.saved_model_path))
        self.model.eval()
        prediction = []
        for test_step, (data, _) in tqdm(enumerate(test), total=len(test), desc=f"test data"):
            data = data.to(self.device)
            output = self.model(data)
            _, predicted = torch.max(output, 1)
            prediction += predicted.tolist()
        return [self.vocab.i2label[i] for i in prediction]

    def accuracy_token_tag(self, predict: List, target: List):
        predict = [self.vocab.i2label[i] for i in predict]
        target = [self.vocab.i2label[i] for i in target]
        all_pred = 0
        correct = 0
        for p, t in zip(predict, target):
            if t == 'O' and p == 'O':
                continue
            all_pred += 1
            if t == p:
                correct += 1
        return (correct / all_pred) * 100

    def dump_test_file(self, test_prediction, test_file_path):
        res = []
        cur_i = 0
        with open(test_file_path) as f:
            lines = f.readlines()
        for line in lines:
            if line == "" or line == "\n":
                res.append(line)
            else:
                pred = f"{line.strip()}{self.vocab.separator}{test_prediction[cur_i]}\n"
                res.append(pred)
                cur_i += 1
        pred_path = f"{self.suffix_run()}.tsv"
        with open(pred_path, mode='w') as f:
            f.writelines(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aligner model')
    parser.add_argument('repr', help="repr need to be {a,b,c,d} when\n"
                                     "a: an embedding vector.\n"
                                     "b: a character-level LSTM.\n"
                                     "c: the embeddings+subword representation used in assignment 2.\n"
                                     "d: a concatenation of (a) and (b) followed by a linear layer.")
    parser.add_argument('trainFile', help="The input file to train on.")
    parser.add_argument('modelFile', help="the file to save the model.")
    parser.add_argument('-o', '--optimizer', type=str, default='AdamW')
    parser.add_argument('-e', '--epochs', help='Number of epochs', default=5, type=int)
    parser.add_argument('-l', '--learning_rate', help='Number of epochs', default=0.001, type=float)
    parser.add_argument('-b', '--batch_size', help='Number of epochs', default=0.001, type=float)
    args = parser.parse_args()

