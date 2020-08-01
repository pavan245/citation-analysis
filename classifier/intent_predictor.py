from typing import Dict, List, Tuple

from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor
from overrides import overrides
from allennlp.models import Model
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models.archival import load_archive
from utils.reader import DataReaderJsonLines, CitationDataSetReader

import os


@Predictor.register('citation_intent_predictor')
class IntentClassificationPredictor(Predictor):
    """
    ~~~Predictor for Citation Intent Classifier~~~

    - This is just a wrapper class around AllenNLP Model
    used for making predictions from the trained/saved model

    """

    def predict(self, text: str, intent: str):
        """
        This function can be called for each data point from the test dataset,
        takes citation text and the target intent as parameters and
        returns output dictionary from :func: `~classifier.nn.BiLstmClassifier.forward` method

        :param text: Citation text from test data
        :param intent: target intent of the data point
        :return: returns output dictionary from Model's forward method
        """
        return self.predict_json({"string": text, "label": intent})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        we get a callback to this method from AllenNLP Predictor,
        passes JsonDict as a parameter with the data that we passed to the prediction_json function earlier.

        And this callback should return the AllenNLP Instance with tokens and target label.

        :param json_dict: json dictionary data with text and intent label
        :return: returns AllenNLP Instance with tokens(ELMo) and target label
        """
        return self._dataset_reader.text_to_instance(json_dict["string"], json_dict["label"])


def make_predictions(model: Model, dataset_reader: DatasetReader, dataset_file_path: str) -> Tuple[list, list]:
    """
    This function takes the pre-trained(saved) Model and DatasetReader(and dataset file path) as arguments
    and returns a Tuple of prediction list and gold/true list.

    - Creates a predictor object with the pre-trained model and dataset reader.
    - Read the data from the passed dataset file path and for each data point, use predictor to predict the intent

    :param model: a trained/saved AllenNLP Model
    :param dataset_reader: Dataset reader object (for tokenizing text and creating Instances)
    :param dataset_file_path: a dataset file path to make predictions

    :return: returns a Tuple of prediction list and true labels list
    """

    # Create predictor class object
    predictor = IntentClassificationPredictor(model, dataset_reader)

    prediction_list = []
    true_list = []

    # read JSON Lines file and Iterate through each datapoint to predict
    jsonl_reader = DataReaderJsonLines(dataset_file_path)
    for citation in jsonl_reader.read():
        true_list.append(citation.intent)
        output = predictor.predict(citation.text, citation.intent)
        prediction_list.append(output['prediction'])

    # returns prediction list and gold labels list - Tuple
    return prediction_list, true_list


def load_model_and_predict_test_data(saved_model_dir: str):
    """

    This function loads the saved model from the specified directory and calls make_predictions function.

    :param saved_model_dir: path of the saved  AllenNLP model (typically from IMS common space)

    :return: returns a list of prediction list and true list
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dev_file_path = project_root + '/data/jsonl/dev.jsonl'
    test_file_path = project_root + '/data/jsonl/test.jsonl'

    # load the archived/saved model
    model_archive = load_archive(os.path.join(saved_model_dir, 'model.tar.gz'))

    # create dataset reader object
    citation_dataset_reader = CitationDataSetReader()

    # make predictions
    y_pred, y_true = make_predictions(model_archive.model, citation_dataset_reader, test_file_path)

    return y_pred, y_true
