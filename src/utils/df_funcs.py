import pandas as pd


def remove_on_multikeys(df1, df2, keys):
    isin = df1.set_index(keys).index.isin(df2.set_index(keys).index)
    return df1.loc[~isin]