"""
Process a batch of annotated tsv files (INCEpTION output).
Save the resulting DataFrame in pkl format.
Update the register of annotated notes ID's so that the notes are excluded from future selections.

USAGE EXAMPLE:
    python process_annotated.py 
        --batch_dir mydirectory/ 
        --outfile mydf.pkl

USAGE EXAMPLE LEGACY STELLA (see naming convention below):
    python process_annotated.py 
        --batch_dir mydirectory/ 
        --outfile mydf.pkl 
        --legacy_parser legacy_stella 

USAGE EXAMPLE LEGACY MARTEN (see naming convention below):
    python process_annotated.py 
        --batch_dir mydirectory/ 
        --outfile mydf.pkl 
        --legacy_parser legacy_marten 
        --path_to_raw ../mydir/raw/notities2017.csv
"""


import argparse
import pandas as pd
from pathlib import Path
from functools import partial


def filename_parser(tsv):
    """
    Parse a filename and return a dict with identifying metadata.

    This parser is used for the naming convention implemented in the a-proof-zonmw project.
    Filename convention:
    'institution--year--MDN--NotitieID--batch.conll'
    Example:
    'vumc--2020--1234567--123456789--batch3.conll'

    Parameters
    ----------
    tsv: Path
        path to tsv file (INCEpTION output)
    
    Returns
    -------
    dict
        dictionary of metadata
    """
    annotator = tsv.stem
    conll = tsv.parent
    institution, year, MDN, NotitieID, batch = conll.stem.split('--')

    return dict(
        annotator = annotator.lower(),
        institution = institution.lower(),
        year = year,
        MDN = MDN,
        NotitieID = NotitieID,
        batch = batch,
        legacy_rawfile = None,
    )


def filename_parser_legacy_stella(tsv):
    """
    Parse a filename and return a dict with identifying metadata.

    This parser is used for the legacy naming convention implemented in the a-proof project for COVID data.
    Filename convention:
    'institution--idx--MDN--NotitieID--NotitieCSN--Notitiedatum--q--search.conll'
    Example:
    'AMC--123--1234567--123456789--123456789--2020-05-18--q1_q2_q3--Search1.conll'

    Parameters
    ----------
    tsv: Path
        path to tsv file (INCEpTION output)
    
    Returns
    -------
    dict
        dictionary of metadata
    """
    annotator = tsv.stem
    conll = tsv.parent
    institution, _, MDN, NotitieID, _, _, _, _ = conll.stem.split('--')

    return dict(
        annotator = annotator.lower(),
        institution = institution.lower(),
        year = '2020',
        MDN = MDN,
        NotitieID = NotitieID,
        batch = 'pilot',
        legacy_rawfile = None,
    )


def filename_parser_legacy_marten(tsv, raw_df):
    """
    Parse a filepath and return a dict with identifying metadata.

    This parser is used for the legacy naming convention implemented in the a-proof project for Non-COVID (2017) data.
    Filename convention:
    'rawfile---idx+1.conll'
    Example:
    'notities_2017_deel2_cleaned.csv---2276.conll'

    Parameters
    ----------
    tsv: Path
        path to tsv file (INCEpTION output)
    raw_df: DataFrame
        original raw data (df read from csv)
    
    Returns
    -------
    dict
        dictionary of metadata
    """
    annotator = tsv.stem
    conll = tsv.parent
    rawfile, idx1 = conll.stem.split('---')
    
    # get NotitieID based on idx+1
    idx = int(idx1) - 1
    NotitieID = raw_df.loc[idx, 'notitieID']
    
    return dict(
        annotator = annotator.lower(),
        institution = 'vumc',
        year = '2017',
        MDN = '',
        NotitieID = NotitieID,
        batch = 'pilot',
        legacy_rawfile = rawfile,
    )


def tsv_to_df(filepath, filename_parser=filename_parser):
    """
    Parse an INCEpTION output file (tsv format) into pd DataFrame.

    Parameters
    ----------
    filepath: Path
        path to tsv file (INCEpTION output)
    filename_parser:
        filename_parser function to use for metadata extraction
    
    Returns
    -------
    DataFrame
        dataframe of annotations from tsv and metadata from filename
    """
    metadata = filename_parser(filepath)
    
    names = ['sen_tok', 'char', 'token', 'label', 'relation']
    return pd.read_csv(
        filepath,
        sep='\t',
        skiprows=5,
        quoting=3,
        names=names
    ).dropna(how='all').query("sen_tok.str[0] != '#'").assign(
        **metadata
    )


def update_annotated_notes_ids(df, fp):
    """
    Update the register of annotated notes ID's with the notes from df.
    If register does not exist, create it.

    Parameters
    ----------
    df: DataFrame
        dataframe of annotations (batch)
    fp: Path
        path to the file that logs annotated notes
    
    Returns
    -------
    None
    """
    cols = ['institution', 'year', 'MDN', 'NotitieID', 'batch', 'legacy_rawfile']
    annotated_notes_ids = df[cols].drop_duplicates()
    print(f"Number of annotated notes in this batch: {annotated_notes_ids.shape[0]}")

    if fp.exists():
        existing_notes_ids = pd.read_csv(fp)
        print(f"Number of notes from previous annotations: {existing_notes_ids.shape[0]}")
        annotated_notes_ids = existing_notes_ids.append(annotated_notes_ids)
    
    annotated_notes_ids.to_csv(fp, index=False)
    print(f"Total number annotated notes: {annotated_notes_ids.shape[0]}")
    print(f"Updated file saved to: {fp}")


def main(batch_dir, outfile, annotfile, legacy_parser=None, path_to_raw=None):
    """
    Process a batch of annotated tsv files (INCEpTION output).
    Save the resulting DataFrame in pkl format.
    Update the register of annotated notes ID's so that the notes are excluded from future selections.

    Parameters
    ----------
    batch_dir: str
        path to a batch of annotation outputs
    outfile: str
        filename for the output pkl
    legacy_parser: str, default=None
        name of legacy parser, if needed
    path_to_raw: str, default=None
        only for legacy_marten; path to the raw data csv file
    
    Returns
    -------
    None
    """
    
    # select filename parser
    global filename_parser
    if legacy_parser == 'legacy_marten':
        raw_df = pd.read_csv(path_to_raw, index_col=0)
        filename_parser = partial(filename_parser_legacy_marten, raw_df=raw_df)
    elif legacy_parser == 'legacy_stella':
        filename_parser = filename_parser_legacy_stella

    # paths
    batch_dir = Path(batch_dir)
    outpath = batch_dir.parent / outfile
    annotfile = Path(annotfile)

    # process tsv files in all subdirectories of batch_dir
    print(f"Processing tsv files in {batch_dir} ...")
    annotated = pd.concat((tsv_to_df(fp, filename_parser) for fp in batch_dir.glob('**/*.tsv')), ignore_index=True)
    print(f"DataFrame created: {annotated.shape=}")

    annotated.to_pickle(outpath)
    print(f"DataFrame saved to {outpath}")

    # save the id's of the annotated notes
    update_annotated_notes_ids(annotated, annotfile)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--batch_dir', default='../../../Non_covid_data_15oct/from_inception_tsv/Inception_Output_Batch1')
    argparser.add_argument('--outfile', default='annotated_df_Batch1_pilot.pkl')
    argparser.add_argument('--annotfile', default='../../data/annotated_notes_ids.csv')
    argparser.add_argument('--legacy_parser', default='legacy_marten')
    argparser.add_argument('--path_to_raw', default='../../../Non_covid_data_15oct/raw/notities_2017_deel2_cleaned.csv')
    args = argparser.parse_args()

    main(args.batch_dir, args.outfile, args.annotfile, args.legacy_parser, args.path_to_raw)


