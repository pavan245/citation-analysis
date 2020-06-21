from classifier.linear_model import MultiClassPerceptron
from utils.csv import read_csv_file
from eval.metrics import f1_score
import utils.constants as const
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_file_path = project_root+'/data/tsv/train.tsv'
test_file_path = project_root+'/data/tsv/test.tsv'

# Read the training dataset
X_train_inst = read_csv_file(train_file_path, '\t')

# set of labels from Training data
labels = set([inst.true_label for inst in X_train_inst])

# Read test data set
X_test_inst = read_csv_file(test_file_path, '\t')

# number of training iterations
epochs = 50

# create MultiClassPerceptron classifier object
clf = MultiClassPerceptron(epochs=epochs, learning_rate=0.5, random_state=101)

# train the model
clf.fit(X_train=X_train_inst, labels=list(labels))


# predict
y_test = clf.predict(X_test_inst)

y_true = [inst.true_label for inst in X_test_inst]

# Model Evaluation
f1_score_micro = f1_score(y_true, y_test, labels, const.AVG_MICRO)
f1_score_macro = f1_score(y_true, y_test, labels, const.AVG_MACRO)
f1_score_none = f1_score(y_true, y_test, labels, None)

# Print F1 Score
for result in f1_score_micro + f1_score_macro + f1_score_none:
    result.print_result()
