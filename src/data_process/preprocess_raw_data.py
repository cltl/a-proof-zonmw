"""
Contains the `raw_to_df` function for combining and preprocessing raw data.
When ran as script will combine and preprocess raw data from:
    - 2017_raw
    - 2018_raw
    - 2020_raw
"""

import json
import re
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
with open(HERE / 'config.data_load.json', 'r', encoding='utf8') as f:
    DATA_LOAD_SETTINGS = json.load(f)


def raw_to_df(datapath):
    """
    Combine and preprocess all raw data files in `datapath` into a single dataframe.
    There are default settings for the `pd.read_csv` function; these defaults may be overwritten by the settings in the `config.data_load.json` (this is necessary due to differences in raw data formats).
    
    Processing steps:
    (1) combine all csv files from the same institution
    (2) drop any duplicated rows
    (3) concatinate all text columns into one string (`all_text`)
    (4) clean up text: remove whitespace and strip string
    (5) convert MDN and NotitieID to str
    (6) add institution
    (7) combine df's per institution into one df

    Parameters
    ----------
    datapath: Path
        path to raw data
    
    Returns
    -------
    DataFrame
        combined and processed dataframe
    """
    datapath = Path(datapath)

    # text cleanup
    regex = re.compile("[\n\r\s\t]+")
    replace_ws = lambda x: regex.sub(' ', str(x))
    combine_columns = lambda s: ' '.join([replace_ws(i).strip() for i in s.values if i==i])

    # settings for read_csv
    defaults = {
        'sep': ';',
        'header': None,
        'encoding': 'utf-8-sig',
    }
    config_from_json = DATA_LOAD_SETTINGS[datapath.stem]

    df = pd.DataFrame(columns=['institution'])
    for institution, settings in config_from_json.items():
        kwargs = {**defaults, **settings}
        glob = kwargs.pop('glob')
        dfs = [pd.read_csv(f, **kwargs) for f in datapath.glob(glob)]
        combined = pd.concat(dfs).drop_duplicates(ignore_index=True)
        combined = combined.iloc[:,:4].assign(
            all_text = combined.iloc[:,4:].apply(combine_columns, axis=1),
            institution = institution)
        combined.columns = ['MDN', 'NotitieID', 'Typenotitie', 'Notitiedatum', 'all_text', 'institution']
        combined = combined.astype({'MDN': str, 'NotitieID': str})
        df = df.append(combined, ignore_index=True)

    print(f"The data directory {datapath.stem} is processed:")
    print(df.institution.value_counts())
    return df


if __name__ == '__main__':

    path = Path('../../data')
    data_dirs = [
        '2017_raw',
        '2018_raw',
        '2020_raw',
    ]

    for datadir in data_dirs:
        df = raw_to_df(path / datadir)
        df.to_pickle(path / datadir / 'processed.pkl')
