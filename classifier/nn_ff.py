"""
Simple feed-forward neural network in PyTorch for baseline results on Scicite data.
Date: July 5th, 2020
"""

import torch
from utils.nn_reader import read_csv_nn,
from utils.nn_reader2 import read_csv_nn_dev
from sklearn.metrics import confusion_matrix
import pandas as pd

class Feedforward(torch.nn.Module):
        """
        Creates and trains a basic feedforward neural network.
        """
        #
        def __init__(self, input_size, hidden_size, output_size):
            """ Sets up all basic elements of NN. """
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.output_size = output_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            #self.relu = torch.nn.ReLU()
            self.tanh = torch.nn.Tanh()
            self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)
            self.sigmoid = torch.nn.Sigmoid()
            self.softmax = torch.nn.Softmax(dim=1)
        #
        def forward(self, x):
            """ Computes output from a given input x. """
            hidden = self.fc1(x)
            #relu = self.relu(hidden)
            tanh = self.tanh(hidden)
            #output = self.fc2(relu)
            output = self.fc2(tanh)
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
    
    l = X_train.shape[0]
    batch_indices = list(zip(list(range(0,l,16))[:-1], list(range(16,l,16))))# + [(l-l%16,l)]
    batch_indices[-1] = (batch_indices[-1][0], l)

    train model
    model.train()
    epochs = 50
    for epoch in range(epochs):
        batch = 0
    	for a,b in batch_indices:
	        optimizer.zero_grad()
	        # forward pass
	        y_pred = model(X_train[a:b])
	        loss = criterion(y_pred, y_train[a:b])
	        #
	        print('Epoch {}, batch {}: train loss: {}'.format(epoch, batch, loss.item()))
	        # backward pass
	        loss.backward()
	        optimizer.step()
	        batch += 1
	    y_pred = model(X_train)
	    loss = criterion(y_pred, y_train)
	    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
	    # shuffle dataset
	    p = torch.randperm(l)
	    X_train = X_train[p]
	    y_train = y_train[p]

    model.eval()
    y_pred = model.forward(X_train)
    after_train = criterion(y_pred, y_train) 
    print('Training loss after training' , after_train.item())

    ## reload the data to get original order
    #X_train, y_train, X_test = read_csv_nn()
    #X_train = torch.FloatTensor(X_train)
    #X_test = torch.FloatTensor(X_test)
    #y_train_ = torch.FloatTensor(y_train)
    #y_train = torch.max(torch.FloatTensor(y_train_),1)[1]
    
    # get dev set to make predictions
    X_dev, y_dev = read_csv_nn_dev()
    X_dev = torch.FloatTensor(X_dev)
    y_dev_pre = torch.FloatTensor(y_dev)
    y_dev = torch.max(torch.FloatTensor(y_dev_pre),1)[1]

    # post-process to get predictions & get back to np format
    y_pred = model.forward(X_dev)
    y_pred_np = y_pred.detach().numpy()
    predmax = np.amax(y_pred_np, axis=1)
    preds = 1*(y_pred_np[:,1]==predmax) + 2*(y_pred_np[:,2]==predmax)
    y_dev_ = y_dev.detach().numpy()
    
    # create confusion matrix
    cm = confusion_matrix(y_dev_, preds)
    print(cm)

    # save predictions
    df = pd.DataFrame()
    df['preds'] = preds
    df['true']  = y_dev_
    probs = y_pred.detach().numpy()
    df['pr0']  = probs[:,0]
    df['pr1']  = probs[:,1]
    df['pr2']  = probs[:,2]
    df['correct'] = df.preds==df.true
    df.to_csv('/Users/iriley/code/machine_learning/lab2020/preds_ffnn.csv')






