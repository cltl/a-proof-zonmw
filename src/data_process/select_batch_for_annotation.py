"""
Function for selecting a batch of notes for annotation.
"""


import pandas as pd


def select_notes(
    data,
    annotators    = ['avelli', 'katsburg', 'meskers', 'opsomer', 'swartjes', 'vervaart', 'ze_edith', 'ze_hinke', 'ze_ron'],
    sources_other = ['2017', '2018', 'non_cov_2020'],
    source_covid  = 'cov_2020',
    n_files       = 60,
    pct_covid     = 0.5,
    pct_kwd       = 0.8,
    domains       = ['ENR', 'ATT', 'STM', 'ADM', 'INS', 'MBW', 'FAC', 'BER', 'ETN'],
    min_matched_domains = 3,
    n_iaa         = 5,
    iaa_sources   = ['cov_2020'],
):
    """
    Select and return a batch of notes from `data` according to the specified parameters.

    Parameters
    ----------
    data: dict
        key is the name of the dataset, value is the dataset (DataFrame)
    annotators: list
        list of the annotators (str)
    sources_other: list
        list of the names of non-covid datasets (as appear in `data` keys)
    source_covid: str
        name of covid dataset (as appears in `data` keys)
    n_files: int
        number of notes per annotator
    pct_covid: float
        desired fraction of covid data in the batch
    pct_kwd: float
        desired fraction of notes containing keywords in the batch
    domains: list
        list of column names in `data` df's that contain the keyword matches
    min_matched_domains: int
        only notes that contain keyword matches from at least this number of ICF domains are selected
    n_iaa: int
        number of files that should be the same for all annotators
    iaa_sources: list
        list of the names of datasets (as appear in `data` keys) from which shared notes are selected

    Returns
    -------
    output: DataFrame
        dataframe with the selected batch 
    """
    output = pd.DataFrame(columns=['NotitieID'])

    # add n_matched_domains to data
    add_matched_domains = lambda df: df[domains].applymap(bool).sum(axis=1)
    data = {source:df.assign(n_matched_domains=add_matched_domains) for source, df in data.items()}

    # assign n_files non-iaa per source
    def get_n_files(source):
        subtract_iaa = n_iaa if source in iaa_sources else 0
        if source == source_covid:
            return int(n_files * pct_covid - subtract_iaa)
        else:
            return int(n_files * (1 - pct_covid) / len(sources_other) - subtract_iaa)
    
    sources = sources_other + [source_covid]
    n_files_per_source = {source:get_n_files(source) for source in sources}

    # select `n_iaa` files for iaa
    selected = pd.DataFrame()
    for source in iaa_sources:
        crit_kwd = data[source].n_matched_domains >= min_matched_domains
        selected = selected.append(data[source].loc[crit_kwd].sample(n_iaa).assign(source=source))

    for annotator in annotators:
        output = output.append(selected.assign(annotator=annotator, samp_meth='kwd_iaa'))

    # create sample per annotator
    query = 'NotitieID not in @output.NotitieID'
    for annotator in annotators:
        for source, n in n_files_per_source.items():
            # keyword sample
            n_kwd = int(n * pct_kwd)
            crit_kwd = data[source].n_matched_domains >= min_matched_domains
            kwd_source = data[source].loc[crit_kwd]
            selected = kwd_source.query(query).sample(n_kwd)
            selected['samp_meth'] = 'kwd'
            selected['source'] = source
            selected['annotator'] = annotator
            output = output.append(selected)

            # random sample
            n_rndm = n - n_kwd
            selected = data[source].query(query).sample(n_rndm)
            selected['samp_meth'] = 'rndm'
            selected['source'] = source
            selected['annotator'] = annotator
            output = output.append(selected)

    return output.reset_index()