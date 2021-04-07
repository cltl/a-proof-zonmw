"""
TBD
"""


import argparse
import json
import random
import re
import pandas as pd


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
    TBD
    """
    cat_dict = categorize_tags(tagset)
    parse_index = create_parse_index(cat_dict)
    parse_label_from_row = lambda row: parse_label(row.label, parse_index)
    
    select_labels = (df.label != '_') & df.label.notna()
    parsed = df.loc[select_labels].apply(parse_label_from_row, result_type='expand', axis=1)
    return df.join(parsed)


def deduplicate_notes(df):
    """
    Some notes are annotated more than once, by different annotators.
    Select one of the annotators randomly per note and keep her/his annotation only.
    """
    choices = df.groupby('NotitieID').annotator.unique().apply(random.choice).reset_index()
    selected = list(choices.values)
    return df.loc[df.set_index(['NotitieID', 'annotator']).index.isin(selected)]


def preprocessing(df, deduplicate=False):
    if deduplicate:
        df = deduplicate_notes(df)
    return df.assign(
        sen_id = lambda df: df.NotitieID.astype(str) + '_' + df.sen_tok.str.split('-').str[0],
        tok = lambda df: df.sen_tok.str.split('-').str[1].astype(int),
    )


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--tagset', default='../../tagsets/tagset.json')
    argparser.add_argument('--infile', default='../../data/from_inception_tsv/annotated_df_sample1.pkl')
    argparser.add_argument('--outfile', default='../../data/from_inception_tsv/annotated_df_sample1_parsed.pkl')
    argparser.add_argument('--deduplicate', dest='deduplicate', action='store_true')
    argparser.set_defaults(deduplicate=False)
    args = argparser.parse_args()

    with open(args.tagset, 'r') as f:
        tagset = json.load(f)['tags']
    
    df = pd.read_pickle(args.infile
    ).pipe(preprocessing, deduplicate=args.deduplicate
    ).pipe(parse_df, tagset)

    df.to_pickle(args.outfile)