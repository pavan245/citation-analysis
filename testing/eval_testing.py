from eval.metrics import f1_score
import utils.constants as const
from sklearn.metrics import f1_score as f1
import os
from utils.csv import read_csv_file

y_true = ['positive', 'positive', 'negative', 'negative', 'positive', 'positive', 'negative', 'negative', 'positive', 'positive', 'negative', 'negative', 'positive', 'positive', 'negative', 'negative']
y_pred = ['positive', 'negative', 'negative', 'positive', 'positive', 'negative', 'negative', 'positive', 'positive', 'negative', 'negative', 'positive', 'positive', 'negative', 'negative', 'negative']

result_list = f1_score(y_true, y_pred, ['positive', 'negative'], const.AVG_MICRO)

for result in result_list:
    result.print_result()

print('SK Learn F1 Score (MICRO):: ', f1(y_true, y_pred, ['positive', 'negative'], average='micro'))

result_list = f1_score(y_true, y_pred, ['positive', 'negative'], const.AVG_MACRO)

for result in result_list:
    result.print_result()

print('SK Learn F1 Score (MACRO):: ', f1(y_true, y_pred, ['positive', 'negative'], average='macro'))


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_file_path = project_root+'/data/tsv/train.tsv'
print(train_file_path)

data = read_csv_file(csv_file_path=train_file_path, delimiter='\t')
for inst in data:
    if len(inst.features) <= 0:
        inst.print()
