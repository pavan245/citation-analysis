from typing import Dict

import numpy as np
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward, Elmo
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from overrides import overrides
from torch.nn import Parameter


@Model.register("basic_bilstm_classifier")
class BiLstmClassifier(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 classifier_feedforward: FeedForward,
                 elmo: Elmo = None,
                 use_input_elmo: bool = False):
        super().__init__(vocab)
        self.elmo = elmo
        self.use_elmo = use_input_elmo
        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("label")
        self.encoder = encoder
        self.classifier_feed_forward = classifier_feedforward
        self.label_accuracy = CategoricalAccuracy()

        self.label_f1_metrics = {}

        for i in range(self.num_classes):
            self.label_f1_metrics[vocab.get_token_from_index(index=i, namespace="label")] = \
                F1Measure(positive_label=i)

        self.loss = torch.nn.CrossEntropyLoss()

        self.attention = Attention(encoder.get_output_dim())

    @overrides
    def forward(self, tokens: Dict[str, torch.LongTensor],
                label: torch.LongTensor) -> Dict[str, torch.LongTensor]:

        input_elmo = None
        elmo_tokens = tokens.pop("elmo", None)

        embedded_text = self.text_field_embedder(tokens)
        text_mask = util.get_text_field_mask(tokens)

        if elmo_tokens is not None:
            tokens["elmo"] = elmo_tokens

            # Create ELMo embeddings if applicable
            if self.elmo:
                if elmo_tokens is not None:
                    elmo_representations = self.elmo(elmo_tokens["elmo_tokens"])["elmo_representations"]
                    if self.use_elmo:
                        input_elmo = elmo_representations.pop()
                    assert not elmo_representations
                else:
                    raise ConfigurationError("Model was built to use Elmo, but input text is not tokenized for Elmo.")

            if self.use_elmo:
                if embedded_text is not None:
                    embedded_text = torch.cat([embedded_text, input_elmo], dim=-1)
                else:
                    embedded_text = input_elmo

        encoded_text = self.encoder(embedded_text, text_mask)

        # Attention
        attn_dist, encoded_text = self.attention(encoded_text, return_attn_distribution=True)

        output_dict = {}
        if label is not None:
            logits = self.classifier_feed_forward(encoded_text)
            class_probabilities = torch.nn.functional.softmax(logits, dim=1)

            output_dict["logits"] = logits

            loss = self.loss(logits, label)
            output_dict["loss"] = loss

            # compute F1 per label
            for i in range(self.num_classes):
                metric = self.label_f1_metrics[self.vocab.get_token_from_index(index=i, namespace="label")]
                metric(class_probabilities, label)
            output_dict['label'] = label

            output_dict['tokens'] = tokens['tokens']

        return output_dict

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        class_probabilities = torch.nn.functional.softmax(output_dict['logits'], dim=-1)
        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        label = [self.vocab.get_token_from_index(x, namespace="label")
                 for x in argmax_indices]
        output_dict['probabilities'] = class_probabilities
        output_dict['positive_label'] = label
        output_dict['prediction'] = label
        citation_text = []
        for batch_text in output_dict['tokens']:
            citation_text.append([self.vocab.get_token_from_index(token_id.item()) for token_id in batch_text])
        output_dict['tokens'] = citation_text

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = {}

        sum_f1 = 0.0
        for name, metric in self.label_f1_metrics.items():
            metric_val = metric.get_metric(reset)
            metric_dict[name + '_P'] = metric_val[0]
            metric_dict[name + '_R'] = metric_val[1]
            metric_dict[name + '_F1'] = metric_val[2]
            if name != 'none':  # do not consider `none` label in averaging F1
                sum_f1 += metric_val[2]

        names = list(self.label_f1_metrics.keys())
        total_len = len(names) if 'none' not in names else len(names) - 1
        average_f1 = sum_f1 / total_len
        metric_dict['average_F1'] = average_f1

        return metric_dict


def new_parameter(*size):
    out = Parameter(torch.FloatTensor(*size))
    torch.nn.init.xavier_normal_(out)
    return out


class Attention(torch.nn.Module):
    """ Simple multiplicative attention"""

    def __init__(self, attention_size):
        super(Attention, self).__init__()
        self.attention = new_parameter(attention_size, 1)

    def forward(self, x_in, reduction_dim=-2, return_attn_distribution=False):
        # calculate attn weights
        attn_score = torch.matmul(x_in, self.attention).squeeze()
        # add one dimension at the end and get a distribution out of scores
        attn_distrib = torch.nn.functional.softmax(attn_score.squeeze(), dim=-1).unsqueeze(-1)
        scored_x = x_in * attn_distrib
        weighted_sum = torch.sum(scored_x, dim=reduction_dim)
        if return_attn_distribution:
            return attn_distrib.reshape(x_in.shape[0], -1), weighted_sum
        else:
            return weighted_sum
