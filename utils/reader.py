from typing import Iterable

import jsonlines
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import SpacyTokenizer
from overrides import overrides

from utils.data import Citation


@DatasetReader.register("citation_dataset_reader") # type for config files
class CitationDataSetReader(DatasetReader):
    def __init__(self):
        super().__init__()
        self.tokenizer = SpacyTokenizer()

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        ds_reader = DataReaderJsonLines(file_path)
        for citation in ds_reader.read():
            yield self.text_to_instance(citation_text=citation.text, intent=citation.intent)

    @overrides
    def text_to_instance(self, citation_text: str,
                         intent: str) -> Instance:
        citation_tokens = self.tokenizer.tokenize(citation_text)
        token_indexers = {"elmo": ELMoTokenCharactersIndexer(),
                          "tokens": SingleIdTokenIndexer()}

        fields = {'tokens': TextField(citation_tokens, token_indexers),
                  'label': LabelField(intent)}

        return Instance(fields)


class DataReaderJsonLines:
    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        for line in jsonlines.open(self.file_path):
            yield read_json_line(line)


def read_json_line(line):
    citation = Citation(
        text=line['string'],
        citing_paper_id=line['citingPaperId'],
        cited_paper_id=line['citedPaperId'],
        section_title=line['sectionName'],
        intent=line['label'],
        citation_id=line['id'])

    return citation
