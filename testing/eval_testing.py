from eval.metrics import f1_score
import utils.constants as const
from sklearn.metrics import f1_score as f1

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
