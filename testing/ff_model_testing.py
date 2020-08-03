import sys
import os
sys.path.append(os.getcwd())
from classifier.nn_ff import FeedForward
from sklearn.metrics import f1_score
from eval.metrics import plot_confusion_matrix, get_confusion_matrix

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

clf = FeedForward(28, 9, 3)
clf.fit()
clf.predict()

# predict
y_test = clf.preds
y_true = clf.y_test

# Model Evaluation
labels = set(['background', 'method', 'result'])
f1_score_micro = f1_score(y_true, y_test, average='micro')
f1_score_macro = f1_score(y_true, y_test, average='macro')

# Print F1 Score
print('F1 score (micro): ', f1_score_micro)
print('F1 score (macro): ', f1_score_macro)

# plot confusion matrix
classdict = {0: 'background', 1: 'method', 2: 'result'}
y_test = [classdict[x] for x in y_test]
y_true = [classdict[x] for x in y_true]
plot_path = project_root + '/plots/confusion_matrix_plot_ff.png'
plot_confusion_matrix(get_confusion_matrix(y_true, y_test), 'Feed-forward NN Classifier (Baseline)', plot_path)


