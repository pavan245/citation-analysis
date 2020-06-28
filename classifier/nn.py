from typing import Dict

import torch
from allennlp.models import Model


@Model.register("basic_bilstm_classifier")
class BiLstmClassifier(Model):

    def forward(self, *inputs) -> Dict[str, torch.Tensor]:
        pass
