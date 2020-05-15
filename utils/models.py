from feature_extraction.features import extract_features_from_text


class DataInstance:
    """
    Model Class for carrying Training and Testing data from tsv/csv file.
    Also carries the extracted features.
    """

    def __init__(self, r_id, text, true_label):
        self.did = r_id
        self.text = text
        self.true_label = true_label
        self.predicted_label = None
        self.features = extract_features_from_text(text)

    def print(self):
        print('\nTrue Label :: ', self.true_label, ' Text :: ', self.text)
        print('Features :: ', self.features)
