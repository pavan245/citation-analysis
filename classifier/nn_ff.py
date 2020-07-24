"""
Simple feed-forward neural network in PyTorch for baseline results on Scicite data.
Date: July 5th, 2020
"""

import torch
from utils.nn_reader import read_csv_nn


class Feedforward(torch.nn.Module):
        """
        Creates and trains a basic feedforward neural network.
        """

        def __init__(self, input_size, hidden_size, output_size):
            """ Sets up all basic elements of NN. """
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.output_size = output_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)
            self.sigmoid = torch.nn.Sigmoid()
            self.softmax = torch.nn.Softmax(dim=1)
        
        def forward(self, x):
            """ Computes output from a given input x. """
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.softmax(output)
            return output



if __name__=='__main__':
    """ Reads in the data, then trains and evaluates the neural network. """

    X_train, y_train, X_test = read_csv_nn()

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train_ = torch.FloatTensor(y_train)
    y_train = torch.max(torch.FloatTensor(y_train_),1)[1]

    model = Feedforward(28, 9, 3)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

    model.eval()
    y_pred = model(X_train)
    before_train = criterion(y_pred, y_train)
    print('Test loss before training' , before_train.item())

    model.train()
    epoch = 2000
    for epoch in range(epoch):
        optimizer.zero_grad()
        # forward pass
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
       
        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        # backward pass
        loss.backward()
        optimizer.step()

    model.eval()
    y_pred = model(X_train)
    after_train = criterion(y_pred, y_train) 
    print('Training loss after training' , after_train.item())



