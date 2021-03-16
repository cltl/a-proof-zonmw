"""
TBD

NOTE: This script requires spaCy's Dutch language pipeline. If you don't have it downloaded, run the command: `python -m spacy download nl_core_news_sm`
"""


def text_to_conll(text, nlp):
    """
    Convert text to CoNLL format, using a language-specific spaCy processing class.

    Parameters
    ----------
    text: str
        text to convert to CoNLL format
    nlp: spacy Language class
        language-specific class that turns text into Doc objects

    Returns
    -------
    conll: str
        string in CoNLL format; each token on a separate line, an empty line between sentences, a column of dummy ent_type next to each token
    """
    text = text.strip()
    doc = nlp(text)

    # add empty line before each sentence start
    add_emptyline = lambda token: '' if token.is_sent_start else None
    tups = [(add_emptyline(token), token.text) for token in doc]

    # flatten tuples into list and drop first empty line
    tokens = [item for token in tups for item in token if item is not None][1:]

    # add a dummy ent_type ("O") value to each token except empty line
    tok_ents = [f"{token} O" if token != '' else '' for token in tokens]

    conll = '\n'.join(tok_ents)

    return conll


def row_to_conllfile(row, nlp, outdir, batch):
    """
    Process a row from a dataframe: the string in the `all_text` column is converted to CoNLL format and is written into a file in outdir.

    Parameters
    ----------
    row: pd Series
        required fields: MDN, NotitieID, NotitieCSN, all_text
    nlp: spacy Language class
        language-specific class that turns text into Doc objects
    outdir: Path
        path to store the output conll file
    hospital: str
        the hospital from which the data originates, to be used in the filename
    year: str
        the year of the data, to be used in the filename
    batch: str
        the batch name, to be used in the filename

    Returns
    -------
    None
    """
    outfile = outdir / f"{row.institution}--{row.Notitiedatum[:4]}--{row.MDN}--{row.NotitieID}--{batch}.conll"
    with open(outfile, 'w', encoding="utf-8") as f:
        f.write(text_to_conll(row.all_text, nlp))