import os
from classifier.linear_model import MultiClassPerceptron
from sklearn.metrics import confusion_matrix as cm
from utils.csv import read_csv_file
from eval.metrics import f1_score
import utils.constants as const
import pandas as pd
import numpy as np


train_file_path = '/Users/iriley/code/citation-analysis/data/tsv/train.tsv'
dev_file_path   = '/Users/iriley/code/citation-analysis/data/tsv/dev.tsv'


# Read the training dataset
X_train_inst = read_csv_file(train_file_path, '\t')

# set of labels from Training data
labels = set([inst.true_label for inst in X_train_inst])

# Read test data set
X_dev_inst = read_csv_file(dev_file_path, '\t')

# number of training iterations
epochs = 50

# create MultiClassPerceptron classifier object
clf = MultiClassPerceptron(epochs=epochs, learning_rate=0.5, random_state=101)

# train the model
clf.fit(X_train=X_train_inst, labels=list(labels))

# predict
y_pred = clf.predict(X_dev_inst)
y_scores = np.array(clf.get_class_scores(X_dev_inst))

y_true = [inst.true_label for inst in X_dev_inst]

labeldict = {'background': 0, 'method': 1, 'result': 2}
y_pred = np.array([labeldict[x] for x in y_pred])
y_true = np.array([labeldict[x] for x in y_true])

conmat = cm(y_true, y_pred)

df = pd.DataFrame()
df['pred'] = y_pred
df['true'] = y_true
df['correct'] = y_pred==y_true
df['score0'] = np.round(y_scores[:,0],3)
df['score1'] = np.round(y_scores[:,1],3)
df['score2'] = np.round(y_scores[:,2],3)

df.to_csv('/Users/iriley/code/machine_learning/lab2020/preds_perceptron.csv', index=False)

## Model Evaluation
#f1_score_micro = f1_score(y_true, y_pred, labels, const.AVG_MICRO)
#f1_score_macro = f1_score(y_true, y_pred, labels, const.AVG_MACRO)
#f1_score_none  = f1_score(y_true, y_pred, labels, None)

## Print F1 Score
#for result in f1_score_micro + f1_score_macro + f1_score_none:
#    result.print_result()