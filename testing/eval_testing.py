from eval.metrics import f1_score
import utils.constants as const

y_true = ['positive', 'positive', 'negative', 'negative', 'positive', 'positive', 'negative', 'negative', 'positive', 'positive', 'negative', 'negative', 'positive', 'positive', 'negative', 'negative']
y_pred = ['positive', 'negative', 'negative', 'positive', 'positive', 'negative', 'negative', 'positive', 'positive', 'negative', 'negative', 'positive', 'positive', 'negative', 'negative', 'negative']

result_list = f1_score(y_true, y_pred, ['positive', 'negative'], const.AVG_MICRO)

for result in result_list:
    result.print_result()

result_list = f1_score(y_true, y_pred, ['positive', 'negative'], const.AVG_MACRO)

for result in result_list:
    result.print_result()
