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
    """"Predictor for Citation Intent Classifier"""

    def predict(self, text: str, intent: str):
        return self.predict_json({"citation_text": text, "intent": intent})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(json_dict["citation_text"], json_dict["intent"])


def make_predictions(model: Model, dataset_reader: DatasetReader, file_path: str) -> Tuple[
    List[Dict[str, float]], list]:
    """Make predictions using the given model and dataset reader"""

    predictor = IntentClassificationPredictor(model, dataset_reader)

    prediction_list = []
    true_list = []

    vocab = model.vocab

    jsonl_reader = DataReaderJsonLines(file_path)
    i = 0
    for citation in jsonl_reader.read():
        i += 1
        true_list.append(citation.intent)
        output = predictor.predict(citation.text, citation.intent)
        prediction_list.append(output['prediction'])
        # prediction_list.append({vocab.get_token_from_index(label_id, 'labels'): prob
        #                         for label_id, prob in enumerate(output['probabilities'])})
        if i == 10:
            break

    return prediction_list, true_list


def load_model_and_run_predictions(saved_model_dir: str):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dev_file_path = project_root + '/data/jsonl/dev.jsonl'
    test_file_path = project_root + '/data/jsonl/test.jsonl'

    model_archive = load_archive(os.path.join(saved_model_dir, 'model.tar.gz'))
    citation_dataset_reader = CitationDataSetReader()

    y_pred, y_true = make_predictions(model_archive.model, citation_dataset_reader, test_file_path)

    print('Predictions ', y_pred)
    print('True Labels ', y_true)
