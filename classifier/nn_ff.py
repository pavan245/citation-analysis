"""
Simple feed-forward neural network in PyTorch for baseline results on Scicite data.
Created: July 5th, 2020
"""

import torch
from utils.nn_reader import read_csv_nn
import numpy as np


class FeedForward(torch.nn.Module):
    """
    Creates and trains a basic feedforward neural network.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """ Sets up all basic elements of NN. """
        super(FeedForward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        """ Computes output from a given input x. """
        hidden = self.fc1(x)
        tanh = self.tanh(hidden)
        output = self.fc2(tanh)
        output = self.softmax(output)
        return output

    def read_data(self):
        """" Reads in training and test data and converts it to proper format. """
        self.X_train_, self.y_train_, self.X_test_ = read_csv_nn()
        yclass = np.array([(x[1] == 1) + 2 * (x[2] == 1) for x in self.y_train_])
        is0 = yclass == 0
        is1 = yclass == 1
        is2 = yclass == 2
        self.X0 = torch.FloatTensor(self.X_train_[is0])
        self.X1 = torch.FloatTensor(self.X_train_[is1])
        self.X2 = torch.FloatTensor(self.X_train_[is2])
        self.y0 = torch.LongTensor(np.zeros((sum(is0),)))
        self.y1 = torch.LongTensor(np.ones((sum(is1),)))
        self.y2 = torch.LongTensor(2 * np.ones((sum(is2),)))
        self.l0 = sum(is0)
        self.l1 = sum(is1)
        self.l2 = sum(is2)

    def fit(self, epochs=100, batch_size=16, lr=0.01, samples0=1000, samples1=1000, samples2=1000):
        """ Trains model, using cross entropy loss and SGD optimizer. """
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr)
        self.samples0 = samples0
        self.samples1 = samples1
        self.samples2 = samples2

        self.eval()  # put into eval mode

        # initialize training data
        self.shuffle()
        y_pred = self.forward(self.X_train)
        before_train = self.criterion(y_pred, self.y_train)
        print('Test loss before training', before_train.item())

        # setup for batches
        l = self.samples0 + self.samples1 + self.samples2  # total length
        batch_indices = list(zip(list(range(0, l, batch_size))[:-1], list(range(16, l, batch_size))))
        batch_indices[-1] = (batch_indices[-1][0], l)

        # train model
        self.train()  # put into training mode
        for epoch in range(epochs):
            batch = 0
            for a, b in batch_indices:
                self.optimizer.zero_grad()

                # forward pass
                y_pred = self.forward(X_train[a:b])
                loss = self.criterion(y_pred, self.y_train[a:b])

                # backward pass
                loss.backward()
                self.optimizer.step()
                batch += 1

            # get loss following epoch
            y_pred = self.forward(self.X_train)
            loss = self.criterion(y_pred, self.y_train)
            print('Epoch {}: train loss: {}'.format(epoch, loss.item()))

            # shuffle dataset
            self.shuffle()

        # display final loss
        self.eval()  # back to eval mode
        y_pred = self.forward(self.X_train)
        after_train = self.criterion(y_pred, self.y_train)
        print('Training loss after training', after_train.item())

    def predict(self):
        """ Generates predictions from model, using test data. """

        # post-process to get predictions & get back to np format
        y_pred = self.forward(self.X_test)
        y_pred_np = y_pred.detach().numpy()
        predmax = np.amax(y_pred_np, axis=1)
        self.preds = 1 * (y_pred_np[:, 1] == predmax) + 2 * (y_pred_np[:, 2] == predmax)
        self.probs = y_pred.detach().numpy()

    def shuffle(self):
        """ Samples and shuffles training data. """

        # create permutations for shuffling
        p0 = torch.randperm(self.l0)
        p1 = torch.randperm(self.l1)
        p2 = torch.randperm(self.l2)
        n = self.l0 + self.l1 + self.l2
        p = torch.randperm(n)

        # sample and shuffle data
        self.X_train = \
        torch.cat((self.X0[p0][:self.samples0], self.X1[p1][:self.samples1], self.X2[p2][:self.samples2]))[p]
        self.y_train = torch.cat((self.y0[:self.samples0], self.y1[:self.samples1], self.y2[:self.samples2]))[p]
