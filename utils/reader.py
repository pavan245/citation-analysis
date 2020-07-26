from typing import Iterable

import jsonlines
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import SpacyTokenizer
from overrides import overrides

from utils.data import Citation


@DatasetReader.register("citation_dataset_reader")  # type for config files
class CitationDataSetReader(DatasetReader):
    """
    We implement this CitationDataSetReader class by subclassing DatasetReader class,
    we also need to override some super class methods

    This CitationDataSetReader class reads the datasets(train|dev|test) and converts them to a collection of Instances.
    We used the default SpacyTokenizer for this project.

    We also need to register this dataset reader, for the Config files to be able to use this class.
    """

    def __init__(self):
        super().__init__()
        # default Spacy Tokenizer
        self.tokenizer = SpacyTokenizer()

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        """

        This function reads the JSON Lines file, tokenize the text for each data point
         and returns a collection of Instances, each instance with tokens and label

        :param file_path: takes the file path as an Argument
        :return: returns a collection of Instances
        """
        ds_reader = DataReaderJsonLines(file_path)
        for citation in ds_reader.read():
            yield self.text_to_instance(citation_text=citation.text, intent=citation.intent)

    @overrides
    def text_to_instance(self, citation_text: str,
                         intent: str) -> Instance:

        """
        :param citation_text: text from the data point
        :param intent: true label of the data instance
        :return: returns Instance class object with tokens & label fields.
        """

        citation_tokens = self.tokenizer.tokenize(citation_text)
        # Use ELMO Token Characters Indexer
        token_indexers = {"elmo": ELMoTokenCharactersIndexer(),
                          "tokens": SingleIdTokenIndexer()}

        fields = {'tokens': TextField(citation_tokens, token_indexers),
                  'label': LabelField(intent)}

        return Instance(fields)


class DataReaderJsonLines:
    """
    Helper class for reading jsonl(JSON Line) files
    """
    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        """
        This method opens the file, reads every line and returns a collection of lines
        :return: collection of Citation Objects, with the required data
        """
        for line in jsonlines.open(self.file_path):
            yield read_json_line(line)


def read_json_line(line):

    """
    :param line: takes the json line dictionary as a parameter
    :return: returns a Citation Object
    """
    citation = Citation(
        text=line['string'],
        citing_paper_id=line['citingPaperId'],
        cited_paper_id=line['citedPaperId'],
        section_title=line['sectionName'],
        intent=line['label'],
        citation_id=line['id'])

    return citation
