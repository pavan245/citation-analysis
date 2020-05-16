"""
Dictionary of Lexicons used for Feature Extraction
"""
ALL_LEXICONS = {

    'COMPARE': ['compar', 'compet', 'evaluat', 'test', 'superior', 'inferior', 'better', 'best', 'good', 'low',
                'worse', 'worst', 'greater', 'larger', 'faster', 'high', 'measur', 'between', 'another', 'similar'],

    'CONTRAST': ['contrast', 'different' 'distinct', 'conflict', 'disagree', 'oppose', 'distinguish', 'contrary'],

    'RESULT': ['estimate', 'evidence', 'experiment', 'find', 'progress', 'observation', 'outcome', 'result', 'performance'],

    'INCREASE': ['increase', 'grow', 'intensify', 'build up', 'explode'],

    'CHANGE': ['adapt', 'adjust', 'augment', 'combine', 'change', 'decrease', 'elaborate', 'expand', 'expand on',
               'extend', 'derive', 'incorporate', 'increase', 'manipulate', 'modify', 'optimize', 'optimise', 'refine',
               'render', 'replace', 'revise', 'substitute', 'tailor', 'upgrade', 'grow'],

    'USE': ['use', 'using', 'apply', 'applied', 'employ', 'make use', 'utilize', 'implement'],

    'PRESENT': ['describe', 'discuss', 'give', 'introduce', 'note', 'notice', 'present', 'propose', 'recapitulate',
                'demonstrate', 'remark', 'report', 'say', 'show', 'sketch', 'state', 'suggest', 'figure'],

    'IMPORTANT': ['important', 'main', 'key', 'basic', 'central', 'crucial', 'critical', 'essential', 'fundamental',
                  'great', 'largest', 'major', 'overall', 'primary', 'principle', 'serious', 'substantial', 'ultimate',
                  'significant', 'remarkable', 'noteworthy', 'crucial', 'emerge'],

    'RESEARCH': ['research', 'paper', 'study', 'studie', 'apply', 'analyze', 'characteri', 'formali', 'investigat',
                 'implement', 'interpret', 'examin', 'observ', 'predict', 'verify', 'work on', 'empirical', 'determin',
                 'experiment', 'exploratory', 'ongoing', 'quantitative', 'qualitative', 'preliminary', 'statistical',
                 'knowledge', 'underway', 'discuss', 'reference', 'publish', 'document', 'orientation'],

    'APPROACH': ['approach', 'account', 'algorithm', 'analys', 'approach', 'application', 'architecture', 'characteri',
                 'component', 'design', 'extension', 'formali', 'framework', 'implement', 'investigat', 'machine',
                 'method', 'methodology', 'module', 'process', 'procedure', 'program', 'prototype', 'strateg',
                 'system', 'technique', 'theory', 'tool', 'treatment'],

    'PUBLIC': ['acknowledge', 'admit', 'agree', 'assert', 'claim', 'complain', 'declare', 'deny', 'explain',
               'hint', 'insist', 'mention', 'proclaim', 'promise', 'protest', 'remark', 'reply', 'report', 'say',
               'suggest', 'swear', 'write'],

    'BEFORE': ['earlier', 'initial', 'past', 'previous', 'prior'],

    'BETTER_SOLUTION': ['boost', 'enhance', 'defeat', 'improve', 'perform better', 'outperform', 'outweigh', 'surpass'],

    'PROFESSIONALS': ['colleagues', 'community', 'computer scientists', 'computational linguists', 'discourse analysts',
                      'expert', 'investigators', 'linguists', 'philosophers', 'psycholinguists',
                      'psychologists', 'researchers', 'scholars', 'semanticists', 'scientists'],

    'MEDICINE': ['medicine', 'tissue', 'gene', 'inflammatory', 'mutant', 'neuro', 'digest', 'ortho', 'kinase', 'pneumonia',
                 'clinical', 'therap', 'kidney', 'receptor', 'cancer', 'synthesis', 'protein', 'syndrom', 'toxin', 'death', 'calcium',
                 'pharma', 'heart', 'disease', 'vitamin', 'tumor', 'blind', 'symptom', 'medical', 'vaccin', 'molecule', 'rna',
                 'biotic', 'patient', 'cells', 'immune', 'blood', 'plasma', 'diagnos', 'neura', 'reproductive', 'plasm', 'drug',
                 'membrane', 'muscle', 'contagious', 'inflam', 'physician', 'dna', 'genome', 'bacteria', 'cavity', 'injury',
                 'antibodies', 'liver', 'treatment', 'pcr', 'acid', 'chronic', 'respirat', 'oxygen', 'stroke', 'antioxidant',
                 'metabolic', 'transmission', 'endogenous', 'syndrome', 'ultrasound', 'pathogen'],

    'MATH': ['matrix', 'gaussian', 'variance', 'radius', 'function', 'comput', 'once', 'twice', 'thrice', 'diagram', 'mean',
             'vector', 'rectangle', 'logic', 'amount', 'maxim', 'minim', 'linear', 'magnitude', 'theorem', 'gradient', 'median',
             'exponential', 'complex', 'graph', 'mean', 'equation', 'offset', 'calculat', 'coefficient', 'discrete', 'equation',
             'math', 'correlation', 'outcome', 'divergence', 'differentiation', 'statistic', 'parameter', 'probabilit', 'multivariate'],

    'COMPUTER_SCIENCE': ['database', 'software', 'evaluation', 'framework', 'computer', 'network', 'algorithm',
                         'dataset','data sets', 'technology', 'kernel', 'metrics', 'nlp', 'xml', 'corpus', 'uml', 'system',
                         'security', 'protocol'],

    'CITATION': ['et al'],  # TODO (for Isaac) :: Write a complex regex for finding Citations in the text

}
