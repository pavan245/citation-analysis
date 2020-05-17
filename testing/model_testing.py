from classifier.linear_model import MultiClassPerceptron
from utils.csv import read_csv_file
from eval.metrics import f1_score
import utils.constants as const
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_file_path = project_root+'/data/tsv/train.tsv'
test_file_path = project_root+'/data/tsv/test.tsv'


X_train_inst = read_csv_file(train_file_path, '\t')

labels = set([inst.true_label for inst in X_train_inst])

X_test_inst = read_csv_file(test_file_path, '\t')

epochs = int(len(X_train_inst)*0.95)

clf = MultiClassPerceptron(epochs=epochs, learning_rate=1, random_state=42)

clf.fit(X_train=X_train_inst, labels=list(labels))

y_test = clf.predict(X_test_inst)

y_true = [inst.true_label for inst in X_test_inst]

f1_score_micro = f1_score(y_true, y_test, labels, const.AVG_MICRO)
f1_score_macro = f1_score(y_true, y_test, labels, const.AVG_MACRO)
f1_score_none = f1_score(y_true, y_test, labels, None)

for result in f1_score_micro+f1_score_macro+f1_score_none:
    result.print_result()
