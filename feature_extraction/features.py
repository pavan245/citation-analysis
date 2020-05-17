import feature_extraction.lexicons as lexicons
from utils.constants import REGEX_CONSTANTS

""" List of supported features for feature extraction from Input String """
FEATURE_LIST = ['COMPARE', 'CONTRAST', 'RESULT', 'INCREASE', 'CHANGE', 'USE', 'PRESENT',
                'IMPORTANT', 'RESEARCH', 'APPROACH', 'PUBLIC', 'BEFORE', 'BETTER_SOLUTION',
                'PROFESSIONALS', 'MEDICINE', 'MATH', 'COMPUTER_SCIENCE', 'CITATION',
                'ACRONYM', 'CONTAINS_YEAR', 'SEQUENCE', 'REFERENCE', 'PERCENTAGE',
                'CONTAINS_URL', 'ENDS_WITH_RIDE', 'ENDS_WITH_RINE', 'ENDS_WITH_ETHYL']

""" Feature Name for Theta Bias -- need to add it to the list of features for all data instances """
THETA_BIAS_FEATURE = 'THETA_BIAS'


def extract_features_from_text(text: str):
    """
    This function takes text string as input, extracts and returns a list of features by checking each word in
        :`~feature_extraction.lexicons.ALL_LEXICONS`
    :param text: takes string text as param
    :return: returns a list of extracted features from the text, empty list for no features
    """

    # ALL_LEXICONS
    lexicon_dict = lexicons.ALL_LEXICONS

    # Initialize the feature list with Theta Bias feature, this feature must be added to all data instances
    text_feature_list = [THETA_BIAS_FEATURE]

    # Iterate through the list features and get list of words from the lexicon dictionary,
    # for each word in the word list, check if it appears in input text and add it to the text feature list
    for feature in FEATURE_LIST:

        # If the feature is Regex Pattern Match, get the pattern from :`~utils.constants.REGEX_CONSTANTS`
        # and match it with the input text
        if feature in REGEX_CONSTANTS:
            pattern = REGEX_CONSTANTS[feature]
            if bool(pattern.search(text)):
                text_feature_list.append(feature)
            continue

        # If the feature is not Regex Pattern Match, then get the list of dictionary words from lexicon dictionary
        word_list = lexicon_dict[feature]
        for word in word_list:
            if word in text.lower():
                text_feature_list.append(feature)
                break

    return text_feature_list
