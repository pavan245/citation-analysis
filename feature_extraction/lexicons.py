"""
Dictionary of Lexicons used for Feature Extraction
"""
ALL_LEXICONS = {

    'COMPARE': ['compar' 'compet', 'evaluat', 'test', 'superior', 'inferior', 'better', 'best', 'worse', 'worst',
                'greater', 'larger', 'faster', 'measur'],

    'CONTRAST': ['contrast', 'different' 'distinct', 'conflict', 'disagree', 'oppose', 'distinguish', 'contrary'],

    'RESULT': ['evidence', 'experiment', 'find', 'progress', 'observation', 'outcome', 'result'],

    'USE': ['use', 'using', 'apply', 'applied', 'employ', 'make use', 'utilize', 'implement'],

    'IMPORTANT': ['important', 'main', 'key', 'basic', 'central', 'crucial', 'critical', 'essential', 'fundamental',
                  'great', 'largest', 'major', 'overall', 'primary', 'principle', 'serious', 'substantial', 'ultimate'],

    'RESEARCH': ['apply', 'analyze', 'characteri', 'formali', 'investigat', 'implement', 'interpret', 'examin',
                 'observ', 'predict', 'verify', 'work on', 'empirical', 'experiment', 'exploratory', 'ongoing',
                 'quantitative', 'qualitative', 'preliminary', 'statistical', 'underway'],

    'APPROACH': ['approach', 'account', 'algorithm', 'analys', 'approach', 'application', 'architecture', 'characteri',
                 'component', 'design', 'extension', 'formali', 'framework', 'implement', 'investigat', 'machine',
                 'method', 'methodology', 'module', 'process', 'procedure', 'program', 'prototype', 'strategy',
                 'system', 'technique', 'theory', 'tool', 'treatment'],

    'PUBLIC': ['acknowledge', 'admit', 'agree', 'assert', 'claim', 'complain', 'declare', 'deny', 'explain',
               'hint', 'insist', 'mention', 'proclaim', 'promise', 'protest', 'remark', 'reply', 'report', 'say',
               'suggest', 'swear', 'write'],
    
    'BEFORE': ['earlier', 'initial', 'past', 'previous', 'prior'],
    
    'BETTER_SOLUTION': ['boost', 'enhance', 'defeat', 'improve', 'perform better', 'outperform', 'outweigh', 'surpass'],

    'PROFESSIONALS': ['colleagues', 'community', 'computer scientists', 'computational linguists', 'discourse analysts',
                      'expert', 'investigators', 'linguists', 'logicians', 'philosophers', 'psycholinguists',
                      'psychologists', 'researchers', 'scholars', 'semanticists', 'scientists'],

    'CITATION': ['et al'],  # TODO (for Isaac) :: Write a complex regex for finding Citations in the text

}
