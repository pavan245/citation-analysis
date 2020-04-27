import utils.constants as const


def f1_score(y_true, y_pred, labels, average):
    if average is None or average == const.AVG_MACRO:
        pr_list = get_precision_recall(y_true, y_pred, labels)
        f1_score_list = []
        f1_sum = 0
        for item in pr_list:
            precision = item['precision']
            recall = item['recall']
            f_score = calculate_f1_score(precision, recall)
            f1_sum += f_score
            if average is None:
                f1_score_list.append(Result(precision, recall, average, item['label'], f_score))

        if average is None:
            return f1_score_list
        elif average == const.AVG_MACRO:
            return [Result(None, None, average, None, f1_sum / len(pr_list))]

    elif average == const.AVG_MICRO:
        pass

    return None


def get_precision_recall(y_true, y_pred, labels=None):
    """
    This method takes Gold Standard Labels and Predicted Labels as arguments
    and computes Precision and Recall for all the labels(including TP, FP, FN).

    Returns a list of dictionaries with precision, recall, tp, fp, fn

    :param y_true: list of Gold labels
    :param y_pred: list of predicted labels
    :param labels: Optional, list of labels for which
    :return: returns the list of dictionaries with Precision and Recall values
                [
                 {'label': 'method', 'precision': 0.71, 'recall': 0.71, 'tp': 5, 'fp': 2, 'fn': 2}
                 {'label': 'background', 'precision': 0.56, 'recall': 0.49, 'tp': 3, 'fp': 2, 'fn': 2}
                ]
    """
    if len(y_true) != len(y_pred):
        raise ValueError('Length of Gold standard labels and Predicted labels must be the same')

    all_labels = False
    if labels is None or len(labels) is 0:
        # get the precision and recall for all the labels
        all_labels = True

    pr_dict = {}

    gold_iter = iter(y_true)
    pred_iter = iter(y_pred)

    while True:
        gold_label = next(gold_iter, None)
        pred_label = next(pred_iter, None)

        # check if the iterator is empty or finished iterating
        if gold_label is None or pred_label is None:
            break

        # Add label entry to the dictionary, if not available
        if gold_label not in pr_dict:
            pr_dict[gold_label] = {'tp': 0, 'fp': 0, 'fn': 0}

        # Add label entry to the dictionary, if not available
        if pred_label not in pr_dict:
            pr_dict[pred_label] = {'tp': 0, 'fp': 0, 'fn': 0}

        if gold_label == pred_label:
            # predicted correctly
            pr_dict[gold_label]['tp'] += 1
        else:
            # Predicted not in class
            pr_dict[gold_label]['fn'] += 1
            # Predicted in class, but Gold is not in class
            pr_dict[pred_label]['fp'] += 1
    # end while

    pr_list = []

    if all_labels:
        labels = list(pr_dict.keys())

    for label in labels:
        tp = pr_dict[label]['tp']
        fp = pr_dict[label]['fp']
        fn = pr_dict[label]['fn']
        precision = get_precision(tp, fp)
        recall = get_recall(tp, fn)
        pr_list.append({'label': label, 'precision': precision, 'recall': recall, 'tp': tp, 'fp': fp, 'fn': fn})

    return pr_list


def get_precision(tp, fp):
    return tp / (tp + fp)


def get_recall(tp, fn):
    return tp / (tp + fn)


def calculate_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


class Result:

    def __init__(self, precision, recall, average, label, f_score):
        self.precision = precision
        self.recall = recall
        self.average = average
        self.label = label
        self.f1_score = f_score

    def print_result(self):
        print('F1 Score :: ', self.f1_score, ' Label :: ', self.label)
