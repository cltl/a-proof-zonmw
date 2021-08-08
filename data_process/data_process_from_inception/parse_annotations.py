"""
Process a pickled DataFrame of annotations so that the labels assigned by the annotators (`label` column) are parsed into separate columns (e.g. `ENR`, `ENR_lvl`, 'background', etc.). Save the resulting parsed DataFrame in pkl format.

The input pickled DataFrame is created from a batch of annotated tsv files with the script `process_annotated.py`.
"""


import argparse
import json
import random
import re
import pandas as pd

import sys
sys.path.insert(0, '..')
from utils.config import PATHS


def categorize_tags(tagset):
    """
    Categorize the tags in `tagset` into: domains, levels, disregard, target, background, plus.

    Parameters
    ----------
    tagset: list
        list of dicts; each dict has a 'tag_name' key

    Returns
    -------
    dict
        dict of categories and the tags belonging to each of them
    """
    tag_names = [i['tag_name'] for i in tagset]
    # define regexes
    rdomain = re.compile('\..*')
    rlevel = re.compile('[a-z]{3}\d')
    rdisregard = re.compile('other_disregard_file')
    rtarget = re.compile('other_target')
    rbackground = re.compile('other_background')
    rplus = re.compile('mbw_plus')
    # find categories
    domains = [tag for tag in tag_names if rdomain.match(tag)]
    levels = [tag for tag in tag_names if rlevel.match(tag)]
    disregard = [tag for tag in tag_names if rdisregard.match(tag)]
    target = [tag for tag in tag_names if rtarget.match(tag)]
    background = [tag for tag in tag_names if rbackground.match(tag)]
    plus = [tag for tag in tag_names if rplus.match(tag)]
    return dict(
        domains=domains,
        levels=levels,
        disregard=disregard,
        target=target,
        background=background,
        plus=plus,
    )


def create_parse_index(cat_dict):
    domains = [i[-3:] for i in cat_dict['domains']]
    levels = [f"{i}_lvl" for i in domains]
    return  pd.Index(domains + levels + ['disregard', 'target', 'background', 'plus'])


def parse_label(label, parse_index):
    s = pd.Series(index=parse_index, dtype=object)
    for idx in s.index:
        if '_lvl' in idx:
            regex = re.compile(f"{idx[:3].lower()}(\d)")
            if regex.search(label):
                s[idx] = int(regex.search(label).group(1))
        else:
            s[idx] = idx in label
    return s


def parse_df(df, tagset):
    """
    Parse the labels assigned by the annotators (`label` column in `df`) so that each type of label (domain / level / background / target / disregard / plus) ends up in a separate column in a df.

    Parameters
    ----------
    df: DataFrame
        dataframe of annotations; the annotations are in the `label` column
    tagset: list
        list of dicts; each dict has a 'tag_name' key

    Returns
    -------
    DataFrame
        dataframe of annotations with the labels parsed into separate columns
    """
    cat_dict = categorize_tags(tagset)
    parse_index = create_parse_index(cat_dict)
    parse_label_from_row = lambda row: parse_label(row.label, parse_index)

    select_labels = (df.label != '_') & df.label.notna()
    parsed = df.loc[select_labels].apply(parse_label_from_row, result_type='expand', axis=1)
    return df.join(parsed)


def deduplicate_notes(df):
    """
    Some notes are annotated more than once, by different annotators (for IAA purposes).
    Select one of the annotators randomly per note and keep her/his annotation only.
    """
    choices = df.groupby('NotitieID').annotator.unique().apply(random.choice).reset_index()
    selected = list(choices.values)
    return df.loc[df.set_index(['NotitieID', 'annotator']).index.isin(selected)]


def preprocessing(df, deduplicate=False):
    """
    Split the `sen_tok` column into sentence ID (combined with note ID) and token ID.
    If the `deduplicate` parameter is True, duplicate notes are removed.
    """
    if deduplicate:
        df = deduplicate_notes(df)
    return df.assign(
        sen_id = lambda df: df.NotitieID.astype(str) + '_' + df.sen_tok.str.split('-').str[0],
        tok = lambda df: df.sen_tok.str.split('-').str[1].astype(int),
    )


def main(tagset, infile, deduplicate, outfile):
    """
    Process a pickled DataFrame of annotations so that the labels assigned by the annotators (`label` column in the DataFrame) are assigned into separate columns (e.g. `ENR`, `ENR_lvl`, etc.).
    Save the resulting DataFrame in pkl format.

    Parameters
    ----------
    tagset: str
        path to a tagset json
    infile: str
        path to the input pkl
    deduplicate: bool
        if True, remove duplicated notes (False for IAA)
    outfile: str
        path to the output pkl

    Returns
    -------
    None
    """

    with open(tagset, 'r') as f:
        tagset = json.load(f)['tags']

    df = pd.read_pickle(infile
    ).pipe(preprocessing, deduplicate=deduplicate
    ).pipe(parse_df, tagset)

    df.to_pickle(outfile)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--rsrcpath', default='resources_inception_config')
    argparser.add_argument('--tagset', default='tagset.json')
    argparser.add_argument('--datapath', default='data_from_inception_tsv')
    argparser.add_argument('--infile', default='annotated_df_sample1.pkl')
    argparser.add_argument('--outfile', default='annotated_df_sample1_parsed.pkl')
    argparser.add_argument('--deduplicate', dest='deduplicate', action='store_true')
    argparser.set_defaults(deduplicate=False)
    args = argparser.parse_args()

    tagset = PATHS.getpath(args.rsrcpath) / args.tagset
    infile = PATHS.getpath(args.datapath) / args.infile
    outfile = PATHS.getpath(args.datapath) / args.outfile

    main(
        tagset,
        infile,
        args.deduplicate,
        outfile,
    )
