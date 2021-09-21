"""
Functions used in pre-processing of data for the machine learning pipelines.
"""


def anonymize(txt, nlp):
    """
    Replace entities of type PERSON and GPE with 'PERSON', 'GPE'.
    Return anonymized text.
    """
    doc = nlp(txt)
    anonym = str(doc)
    to_repl = {str(ent):ent.label_ for ent in doc.ents if ent.label_ in ['PERSON', 'GPE']}
    for string, replacement in to_repl.items():
        anonym = anonym.replace(string, replacement)
    return anonym
