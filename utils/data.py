class Citation(object):
    """ Class representing a citation object """

    def __init__(self,
                 text,
                 citing_paper_id,
                 cited_paper_id,
                 section_title=None,
                 intent=None,
                 citation_id=None
                 ):
        self.text = text
        self.citing_paper_id = citing_paper_id
        self.cited_paper_id = cited_paper_id
        self.section_title = section_title
        self.intent = intent
        self.citation_id = citation_id
