"""
TBD

NOTE: This script requires spaCy's Dutch language pipeline. If you don't have it downloaded, run the command: `python -m spacy download nl_core_news_sm`
"""


import argparse
import re
import spacy
import pandas as pd
from pathlib import Path


def get_regex(keyword, template):
    """
    Turn a keyword into a regex, according to a template id:
    - template 0 is for stems
    - template 1 is for one-word expressions
    - template 2 is for two-word expressions; the two words can appear in the sentence in different order with optional other words in between them

    Parameters
    ----------
    keyword: str
    template: int

    Returns
    -------
    regex: str
    """
    if template == 0:
        return rf"\b{keyword}.+\b"
    elif template == 1:
        return rf"\b{keyword}\b"
    elif template == 2:
        kwd1, kwd2 = keyword.split(' ')
        return rf"\b{kwd1}\b[^.]*?\b{kwd2}\b|\b{kwd2}\b[^.]*?\b{kwd1}\b"


def get_reg_dict(kwd_df):
    """
    Join all regexes belonging to one domain into one regex.

    Parameters
    ----------
    kwd_df: DataFrame
        dataframe with the columns `domain` and `regex`

    Returns
    -------
    dict
        dictionary with the domain as key and a "joined domain regex" as value
    """
    reg_dict = dict()
    for domain in kwd_df.domain.unique():
        reg_dict[domain] = '|'.join(kwd_df.query("domain == @domain").regex)
    return reg_dict


def find_keywords(df, reg_dict):
    """
    Per domain in `reg_dict`, find all matches for the "joined domain regex" in the `all_text` column of the dataframe. Store the matches in a column with the domain name.

    Parameters
    ----------
    df: DataFrame
        dataframe with the column `all_text`
    reg_dict: dict
        dictionary with domains as keys and a "joined domain regex" as values

    Returns
    -------
    DataFrame
        dataframe with the new columns containing matches
    """
    for k, v in reg_dict.items():
        df[k] = df.all_text.str.findall(v, flags=re.IGNORECASE)
    return df


def save_kwd_results(df, domains, outfile):
    """
    Save to csv the ID's of notes in which keywords were found.
    The csv contains a column per domain with a list of the keywords found in a note.

    Parameters
    ----------
    df: DataFrame
        dataframe with notes and results of keyword search
    domains: list
        list of domains, as they appear in the column names of the df
    outfile: str
        path to csv file to write the results

    Returns
    -------
    None
    """
    df.loc[df[domains].apply(any, axis=1)][['MDN', 'NotitieID', 'NotitieCSN'] + domains]
    df.to_csv(outfile, index_label='idx_source_file')
    return None


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


def row_to_conllfile(row, nlp, outdir, hospital, batch):
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
    batch: str
        the batch name, to be used in the filename

    Returns
    -------
    None
    """
    outfile = outdir / f"{hospital}--{row.MDN}--{row.NotitieID}--{row.NotitieCSN}--{batch}.conll"
    with open(outfile, 'w', encoding="utf-8") as f:
        f.write(text_to_conll(row.all_text, nlp))


def main():
    pass

    # STEPS OVERVIEW:
    # ==============
    # load file, combine text columns
    # TO DO: exclude annotated
    # load keywords and get_regex
    # get_reg_dict
    # find_keywords
    # store search results in csv (args.keywords_results)
    # TO DO: select sample
    # convert sample to conll


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--datapath', default='../../../Covid_data_11nov/raw')
    argparser.add_argument('--outpath', default='../../../Covid_data_11nov/to_inception_conll')
    argparser.add_argument('--keywords', default='../../keywords/keywords.xlsx')
    argparser.add_argument('--keywords_results', default='')
    args = argparser.parse_args()

    main()
