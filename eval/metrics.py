def f1_score(y_true, y_pred, labels, average):
    return 0


class Result:

    def __init__(self, precision, recall, average, label):
        self.precision = precision
        self.recall = recall
        self.average = average
        self.label = label
        self.f1_score = 2 * (precision * recall) / (precision + recall)

    def print_result(self):
        print('F1 Score :: ',self.f1_score)
