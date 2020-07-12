from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor
from overrides import overrides


@Predictor.register('citation_intent_predictor')
class IntentClassificationPredictor(Predictor):
    """"Predictor for Citation Intent Classifier"""

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        pass

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        pass
