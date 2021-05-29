from typing import List, Optional
import torch.nn as nn
from vocab import Vocab, SeqVocab
import torch
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=-100)

    return xx_pad, yy_pad, x_lens, y_lens


class Trainer:
    def __init__(self, model: nn.Module, train_data: Dataset, dev_data: Dataset, vocab: Vocab,char_vocab: Vocab, n_ep=1,
                 optimizer='AdamW', train_batch_size=32, steps_to_eval=2500, lr=0.01, part=None,
                 output_path=None):
        # TODO Load for testing need to make surwe part 1 and 2 would still work.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.part = part
        self.model = model
        self.dev_batch_size = 128
        self.vocab = vocab
        self.label_weight = self.get_label_weight(train_data)
        self.train_data = DataLoader(train_data, batch_size=train_batch_size, collate_fn=pad_collate)
        self.dev_data = DataLoader(dev_data, batch_size=self.dev_batch_size,  collate_fn=pad_collate)
        self.char_vocab = char_vocab
        if optimizer == "SGD":
            self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
        elif optimizer == "AdamW":
            self.optimizer = optim.AdamW(model.parameters(),  lr=lr)
        elif optimizer == "Adam":
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            self.optimizer = optim.AdamW(model.parameters())
            print("optimizer supports SGD, Adam, AdamW, Using by Default AdamW")

        self.steps_to_eval = steps_to_eval
        self.n_epochs = n_ep
        self.loss_func = nn.CrossEntropyLoss(weight=self.label_weight)
        self.model.to(self.device)
        lstm_dim = None
        try:
            lstm_dim = self.model.lstm_hidden_dim
        except Exception:
            pass
        self.model_args = {"part": self.part, "task": self.vocab.task, "lr": lr, "epoch": self.n_epochs,
                           "batch_size": train_batch_size ,"hidden_dim": self.model.hidden_dim, "lstm_dim": lstm_dim}
        if output_path is None:
            output_path = self.suffix_run()

        self.saved_model_path = f"{output_path}.bin"

        self.writer = SummaryWriter(log_dir=f"tensor_board/{output_path}")
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
            for step, (data, target, data_lens, target_lens) in tqdm(enumerate(self.train_data), total=len(self.train_data)):
                data = data.to(self.device)
                target = target.to(self.device)
                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                self.model.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data, data_lens)  # Eemnded Data Tensor size (1,5)
                # calculate the loss

                loss = self.loss_func(output, target.view(-1))
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                # update running training loss
                train_loss += loss.item() * data.size(0)
                step_loss += loss.item() * data.size(0)
                if (step+1)*self.train_data.batch_size % self.steps_to_eval == 0:
                    print(f"in step: {(step+1)*self.train_data.batch_size} train loss: {step_loss}")
                    self.writer.add_scalar('Loss/train_step', step_loss, step * (epoch + 1))
                    step_loss = 0.0
                    self.evaluate_model((step+1)*self.train_data.batch_size *(epoch+1), "step")
            print(f"in epoch: {epoch + 1} train loss: {train_loss}")
            self.writer.add_scalar('Loss/train', train_loss, epoch+1)
            self.evaluate_model(epoch, "epoch")

    def evaluate_model(self, step, stage):
        with torch.no_grad():
            self.model.eval()
            loss = 0

            prediction = []
            all_target = []
            for eval_step, (data, target, data_lens, target_lens) in tqdm(enumerate(self.dev_data), total=len(self.dev_data),
                                                  desc=f"dev step {step} loop"):
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data, data_lens)  # Eemnded Data Tensor size (1,5)

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


    def evaluate_data_set(self, data, stage):
        with torch.no_grad():
            self.model.eval()
            loss = 0

            prediction = []
            all_target = []
            for eval_step, (data, target, data_lens, target_lens) in tqdm(enumerate(data), total=len(data),
                                                                          desc=f"test data set"):
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data, data_lens)  # Eemnded Data Tensor size (1,5)

                loss = self.loss_func(output, target.view(-1))
                loss += loss.item() * data.size(0)
                _, predicted = torch.max(output, 1)
                prediction += predicted.tolist()
                all_target += target.view(-1).tolist()
            accuracy = self.accuracy_token_tag(prediction, all_target)
            print(f'Accuracy/dev_{stage}: {accuracy}')
            self.writer.add_scalar(f'Accuracy/dev_{stage}', accuracy, 0)
            self.writer.add_scalar(f'Loss/dev_{stage}', loss, 0)


    def suffix_run(self):
        res = ""
        for k, v in self.model_args.items():
            res += f"{k}_{v}_"
        res = res.strip("_")
        return res

    def test(self, test_df):
        test = DataLoader(test_df, batch_size=self.dev_batch_size,  collate_fn=pad_collate)
        self.model.load_state_dict(torch.load(self.saved_model_path))
        self.model.eval()
        prediction = []
        for eval_step, (data, _, data_lens, _) in tqdm(enumerate(test), total=len(test),
                                                                      desc=f"test data"):
            data = data.to(self.device)
            output = self.model(data, data_lens)
            _, predicted = torch.max(output, 1)
            prediction += predicted.tolist()
        return [self.vocab.i2label[i] for i in prediction]

    def accuracy_token_tag(self, predict: List, target: List):
        predict, target = self.get_unpadded_samples(predict, target)

        all_pred = 0
        correct = 0
        for p, t in zip(predict, target):
            if t == 'O' and p == 'O':
                continue
            all_pred += 1
            if t == p:
                correct += 1
        return (correct / all_pred) * 100

    def get_unpadded_samples(self, predict, target):
        no_pad_predict = []
        no_pad_target = []

        for p, t in zip(predict, target):
            if t == -100:
                continue
            no_pad_predict.append(self.vocab.i2label[p])
            no_pad_target.append(self.vocab.i2label[t])
        return no_pad_predict, no_pad_target

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

    def get_label_weight(self, train_data):
        try:
            all_labels = np.array([self.vocab.label2i[item] for sublist in [t.labels for t in train_data.data] for item in sublist])
            classes=np.unique(all_labels)
            cw = compute_class_weight('balanced', classes=classes, y=all_labels)
            res = torch.Tensor(cw)
            res.to(self.device)
            return res

        except Exception as e:
            res = torch.Tensor([0.5, 0.5])
            res.to(self.device)
            return res


