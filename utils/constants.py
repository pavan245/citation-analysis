import re


AVG_MICRO = 'MICRO'
AVG_MACRO = 'MACRO'

REGEX_CONSTANTS = {

    # Regex for matching Acronym Patterns -> COVID-19 / SEKA / SMY2 / EAP1 / SCP16 / ASC1 / DENV-2
    'ACRONYM': re.compile(r"[m0-9\W^]([A-Z]{2,})[s\.,:\-$]"),

    # Regex for matching Years in the text - > 1995 / 2020 / 2019
    'CONTAINS_YEAR': re.compile(r"(?<=[^0-9])1[8-9][0-9]{2}(?=[^0-9$])|(?<=[^0-9])20[0-2][0-9](?=[^0-9$])"),

    # Regex for matching Number Sequences in the text -> (15) / (10, 11, 112, 113) / (1,7,8,10-14)
    'SEQUENCE': re.compile(r"\([\d.*\)"),

    # Regex for matching References in the text -> [4] / [ 10-17, 19, 20] / [123, 500]
    'REFERENCE': re.compile(r"\[\d.*\]"),

    # Regex for matching percentages in the text -> 99% / 99.99% / 10 % / 23.98% / 10-20% / 25%-30%
    'PERCENTAGE': re.compile(r"\d[\d\.\-]+%"),

    # Regex for matching URLs -> http://www.phrap.org/, http://www. , http://carcfordjournals. ,
    # https://www.ims.uni-stuttgart.de/
    'URL': re.compile(r"https?://\S+")#...\S+(?=\.?,?:?[\s\"$])")
}
