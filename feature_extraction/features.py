from utils.constants import REGEX_CONSTANTS
import feature_extraction.lexicons as lexicons
import re

""" List of supported features for feature extraction from Input String """
FEATURE_LIST = ['COMPARE', 'CONTRAST', 'RESULT', 'USE', 'IMPORTANT', 'RESEARCH', 'APPROACH',
                'PUBLIC', 'BEFORE', 'BETTER_SOLUTION', 'CITATION', 'ACRONYM']

REGEX_FEATURES = ['ACRONYM']


def extract_features_from_text(text: str):
    """
    This function takes text string as input, extracts and returns a list of features by checking each word in
        :`~feature_extraction.lexicons.ALL_LEXICONS`
    :param text: takes string text as param
    :return: returns a list of extracted features from the text, empty list for no features
    """

    # ALL_LEXICONS
    lexicon_dict = lexicons.ALL_LEXICONS

    text_feature_list = []
    # Iterate through the list features and get list of words from the lexicon dictionary,
    # for each word in the word list, check if it appears in input text and add it to the text feature list
    for feature in FEATURE_LIST:
        if feature in REGEX_FEATURES:
            regex = REGEX_CONSTANTS[feature]
            if bool(re.match(regex, text)):
                text_feature_list.append(feature)
            continue
        word_list = lexicon_dict[feature]
        for word in word_list:
            if word in text:
                text_feature_list.append(feature)
                break

    return text_feature_list
