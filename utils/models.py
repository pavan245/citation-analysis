
class DataInstance:
    """
    Model Class for carrying Training and Testing data from tsc/csv file
    """

    def __init__(self, r_id, text, true_label):
        self.did = r_id
        self.text = text
        self.true_label = true_label

    def print(self):
        print('True Label :: ', self.true_label, ' Text :: ', self.text)
