"""
Functions for finding keyword matches in a dataframe with text and storing the results in a pkl file.
When ran as script will find and store keyword matches for the `processed.pkl` files in:
    - 2017_raw
    - 2018_raw
    - 2020_raw

The version of keywords can be given as a parameter to the script, for example:
    python keyword_search.py --kwd_v v1
"""


import argparse
import re
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
        return rf"\b{keyword}.*?\b"
    elif template == 1:
        return rf"\b{keyword}\b"
    elif template == 2:
        kwd1, kwd2 = keyword.split(' ')
        return rf"\b{kwd1}\b[^.]*?\b{kwd2}\b|\b{kwd2}\b[^.]*?\b{kwd1}\b"


def get_reg_dict(kwd_df):
    """
    Create a dictionary of domains. All regexes belonging to a domain are joined into one regex.

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
    Save to pkl the ID's of notes in which keywords were found.
    The pickled dataframe contains a column per domain with a list of the keywords found in a note.

    Parameters
    ----------
    df: DataFrame
        dataframe with notes and results of keyword search
    domains: list
        list of domains, as they appear in the column names of the df
    outfile: str
        path to pkl file to write the results

    Returns
    -------
    None
    """
    matched_rows = df[domains].apply(any, axis=1)
    df = df.loc[matched_rows][['institution', 'MDN', 'NotitieID'] + domains]
    df.to_pickle(outfile)
    print(f"Results {len(df)=} are saved to {outfile}")
    return None


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--kwd_v', default='v1')
    args = argparser.parse_args()

    path = Path('../../data')
    data_dirs = [
        '2017_raw',
        '2018_raw',
        '2020_raw',
    ]

    kwd_path = f'../../keywords/keywords_{args.kwd_v}.xlsx'
    kwd = pd.read_excel(kwd_path)
    kwd['regex'] = kwd.apply(lambda row: get_regex(row.keyword, row.regex_template_id), axis=1)
    reg_dict = get_reg_dict(kwd)

    domains = ['ENR', 'ATT', 'STM', 'ADM', 'INS', 'MBW', 'FAC', 'BER']

    for datadir in data_dirs:
        infile = path / datadir / 'processed.pkl'
        outfile = path / f"keyword_results/{(path / datadir).stem[:4]}_kwd_{args.kwd_v}.pkl"
        
        df = pd.read_pickle(infile)
        print(f"Processing {datadir}: {len(df)=}")
        df = find_keywords(df, reg_dict)
        save_kwd_results(df, domains, outfile)
        

