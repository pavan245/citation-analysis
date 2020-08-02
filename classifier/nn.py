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

    """
    Two things to note first:
        - This BiLstmClassifier is a subclass of AllenNLP's Model class
        - This class registers the type "basic_bilstm_classifier"  using @Model.register() decorator,
            this is required for the Config file to identify the Model class.

    AllenNLP Model is similar to PyTorch Module, it implements forward() method and returns an output dictionary
    with loss, logits and more....

    The constructor parameters should match with configuration in the config file, the Vocabulary is composed by
    the library or train pipeline after reading data using Dataset Reader.

     In this model, we used Elmo embeddings, 1 layer BiLSTM (encoder) and 2 Feed-forward layers.
     The train command/pipeline calls the forward method for a batch of Instances,
     and the forward method returns the output dictionary with loss, logits, label and F1 metrics

    """

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
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.encoder = encoder
        self.classifier_feed_forward = classifier_feedforward
        self.label_accuracy = CategoricalAccuracy()

        self.label_f1_metrics = {}

        # create F1 Measures for each class
        for i in range(self.num_classes):
            self.label_f1_metrics[vocab.get_token_from_index(index=i, namespace="labels")] = \
                F1Measure(positive_label=i)

        self.loss = torch.nn.CrossEntropyLoss()

        self.attention = Attention(encoder.get_output_dim())

    @overrides
    def forward(self, tokens: Dict[str, torch.LongTensor],
                label: torch.LongTensor) -> Dict[str, torch.LongTensor]:

        """
        The training loop takes a batch of Instances and passes it to the forward method

        :param tokens: tokens from the Instance
        :param label: label from the data Instance

        :return: returns an output dictionary after forwarding inputs to the model
        """

        input_elmo = None
        # pop the "elmo" key and add it later
        elmo_tokens = tokens.pop("elmo", None)

        embedded_text = self.text_field_embedder(tokens)
        text_mask = util.get_text_field_mask(tokens)

        if elmo_tokens is not None:
            tokens["elmo"] = elmo_tokens

            # Create ELMo embeddings if applicable
            if self.elmo:
                if elmo_tokens is not None:
                    # get elmo representations from Tokens
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

        # pass the embedded text to the LSTM encoder
        encoded_text = self.encoder(embedded_text, text_mask)

        # Attention
        attn_dist, encoded_text = self.attention(encoded_text, return_attn_distribution=True)

        output_dict = {}
        if label is not None:
            logits = self.classifier_feed_forward(encoded_text)

            # Probabilities from Softmax
            class_probabilities = torch.nn.functional.softmax(logits, dim=1)

            output_dict["logits"] = logits

            # loss calculation
            loss = self.loss(logits, label)
            output_dict["loss"] = loss

            # compute F1 per label
            for i in range(self.num_classes):
                metric = self.label_f1_metrics[self.vocab.get_token_from_index(index=i, namespace="labels")]
                metric(class_probabilities, label)
            output_dict['label'] = label

            output_dict['tokens'] = tokens['tokens']

        return output_dict

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """

        The predict command/pipeline calls this method with the output dictionary from forward() method.

        The returned output dictionary will also be printed in the console when the predict command is executed

        :param output_dict: output dictionary
        :return: returns human readable output dictionary
        """
        class_probabilities = torch.nn.functional.softmax(output_dict['logits'], dim=-1)
        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)

        # get the label from vocabulary
        label = [self.vocab.get_token_from_index(x, namespace="labels")
                 for x in argmax_indices]
        output_dict['probabilities'] = class_probabilities
        output_dict['positive_label'] = label
        output_dict['prediction'] = label

        # return ouput dictionary
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        """

        This method gets a call from the train pipeline,
        and the returned metrics dictionary will be printed in the Console while Training.

        The returned metrics dictionary contains class-wise F1 Scores, Average F1 score and loss

        :param reset: boolean

        :return: returns a metrics dictionary with Class Level F1 scores and losses
        """

        metric_dict = {}

        sum_f1 = 0.0
        for name, metric in self.label_f1_metrics.items():
            metric_val = metric.get_metric(reset)
            metric_dict[name + '_F1'] = metric_val[2]
            if name != 'none':  # do not consider `none` label in averaging F1
                sum_f1 += metric_val[2]

        names = list(self.label_f1_metrics.keys())
        total_len = len(names) if 'none' not in names else len(names) - 1
        average_f1 = sum_f1 / total_len
        metric_dict['AVG_F1_Score'] = average_f1

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
