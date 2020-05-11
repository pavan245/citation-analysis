import re


AVG_MICRO = 'MICRO'
AVG_MACRO = 'MACRO'

REGEX_CONSTANTS = {
    'ACRONYM': re.compile(r"\s*\b[A-Z.]{2,}s?\b\s*"),
    'CONTAINS_YEAR': re.compile('.*([1-2][0-9]{3})'),
    'SEQUENCE': re.compile(r'\s+\((\d+,* *)*\)\s+'),
    'REFERENCE': re.compile(r"\s*\[(\d+,* *)*\]\s*")
}
