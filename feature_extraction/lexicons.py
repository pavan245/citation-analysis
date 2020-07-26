"""
Dictionary of Lexicons used for Feature Extraction
"""
ALL_LEXICONS = {

    'COMPARE': ['compar', 'compet', 'evaluat', 'test', 'superior', 'inferior', 'better', 'best', 'good', 'low',
                'wors', 'great', 'larger', 'faster', 'high', 'measur', 'between', 'another', 'similar'],

    'CONTRAST': ['contrast', 'different' 'distinct', 'conflict', 'disagree', 'oppose', 'distinguish', 'contrary'],

    'RESULT': ['estimate', 'evidence', 'experiment', 'find', 'progress', 'observation', 'outcome', 'result', 'performance'],

    'INCREASE': ['increase', 'grow', 'intensify', 'build up', 'explode'],

    'CHANGE': ['adapt', 'adjust', 'augment', 'combine', 'change', 'decrease', 'elaborate', 'expand', 'expand on',
               'extend', 'derive', 'incorporate', 'increase', 'manipulate', 'modify', 'optimize', 'optimise', 'refine',
               'render', 'replace', 'revise', 'substitute', 'tailor', 'upgrade', 'grow'],

    'USE': ['use', 'using', 'apply', 'applied', 'employ', 'make use', 'utilize', 'implement'],

    'PRESENT': ['describe', 'discuss', 'give', 'introduce', 'note', 'notice', 'present', 'propose', 'recapitulate',
                'demonstrate', 'remark', 'report', 'say', 'show', 'sketch', 'state', 'suggest', 'figure', 'indicate',
                'specify', 'explain'],

    'IMPORTANT': ['important', 'main', 'key', 'basic', 'central', 'crucial', 'critical', 'essential', 'fundamental',
                  'great', 'largest', 'major', 'overall', 'primary', 'principle', 'serious', 'substantial', 'ultimate',
                  'significant', 'remarkable', 'noteworthy', 'crucial', 'emerge'],

    'RESEARCH': ['research', 'paper', 'study', 'studie', 'apply', 'analyze', 'characteri', 'formali', 'investigat',
                 'implement', 'interpret', 'examin', 'observ', 'predict', 'verify', 'work on', 'empirical', 'determin',
                 'experiment', 'exploratory', 'ongoing', 'quantitative', 'qualitative', 'preliminary', 'statistical',
                 'knowledge', 'underway', 'discuss', 'reference', 'publish', 'document', 'orientation',
                 'literature', 'experience'],

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

    'MEDICINE': ['medicine', 'tissue', 'gene', 'inflammatory', 'mutant', 'neuro', 'digest', 'ortho', 'kinase',
                 'clinical', 'therap', 'kidney', 'receptor', 'cancer', 'synthesis', 'protein', 'syndrom', 'toxin', 'death',
                 'pharma', 'heart', 'disease', 'vitamin', 'tumor', 'blind', 'symptom', 'medical', 'vaccin', 'molecule',
                 'biotic', 'patient', 'cells', 'immune', 'blood', 'plasma', 'diagnos', 'neura', 'reproductive', 'plasm', 'drug',
                 'membrane', 'muscle', 'contagious', 'inflam', 'physician', 'dna', 'genome', 'bacteria', 'cavity', 'injury',
                 'antibodies', 'liver', 'treatment', 'pcr', 'acid', 'chronic', 'respirat', 'oxygen', 'stroke', 'antioxidant', 'obesity',
                 'metabolic', 'transmission', 'endogenous', 'syndrome', 'ultrasound', 'pathogen', 'inject', 'laparoscop',
                 'circulat', 'ventricle', 'tract', 'pneumonia', 'calcium',  'rna', 'organism', 'biolog', 'x-ray'],

    'MATH': ['matrix', 'gaussian', 'variance', 'radius', 'function', 'comput', 'once', 'twice', 'thrice', 'diagram', 'mean',
             'vector', 'rectangle', 'logic', 'amount', 'maxim', 'minim', 'linear', 'magnitude', 'theorem', 'gradient', 'median',
             'exponential', 'complex', 'graph', 'mean', 'equation', 'offset', 'calculat', 'coefficient', 'discrete', 'equation',
             'frequen', 'math', 'correlation', 'outcome', 'divergence', 'differentiation', 'statistic', 'parameter',
             'probabilit', 'multivariate', 'negative', 'positive', 'regression', 'digit'],

    'COMPUTER_SCIENCE': ['database', 'software', 'evaluation', 'framework', 'computer', 'network',
                         'algorithm', 'dataset','data sets', 'technology', 'kernel', 'metrics', 'nlp', 'xml',
                         'corpus', 'uml', 'system', 'security', 'protocol', 'classification', 'data transform',
                         'memory', 'java', 'python', 'cluster', 'epoch', 'training', 'deadlock', 'technique'],

    'CITATION': ['et al']

}
