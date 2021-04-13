import pandas as pd
from string import Template
from textwrap import indent


TABLES = []


def show_latex(
    df,
    caption,
    label,
    column_format=None,
    cell_format=lambda x: f'{x:,}',
):
    """
    Generate a LaTeX table from the `df` and append it to TABLES.

    Parameters
    ----------
    df: DataFrame
        dataframe to be turned into a latex table
    caption: str
        caption for the latex table
    label: str
        label for the latex table
    column_format: str, default=None
        column format to overwrite the default one in the function
    cell_format: func
        function to convert `df` values (by default to str)
    
    Returns
    -------
    df: DataFrame
    """
    template = Template('\n'.join([
        r"\begin{table}[]",
        r"    \centering",
        r"$tabular",
        r"    \caption{$caption}",
        r"    \label{tab:$label}",
        r"\end{table}",
    ]))
    alignment = {'int64': 'r', 'float64': 'r'}
    if column_format is None:
        col_formats = [alignment.get(str(i), 'l') for i in df.dtypes.values]
        idx_formats = ['l'] * df.index.nlevels
        column_format = ''.join(idx_formats + col_formats)
    df = df.applymap(cell_format)
    tab = ' ' * 4
    tabular = indent(df.to_latex(column_format=column_format), tab).rstrip('\n')
    table = template.substitute(tabular=tabular, caption=caption, label=label)
    TABLES.append(table)
    return df


def add_colname(df, label):
    """
    Add `label` to column names as an extra level.

    Parameters
    ----------
    df: DataFrame
    label: str
    
    Returns
    -------
    df: DataFrame
        dataframe with MultiIndex columns
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return pd.concat([df], keys=[label], axis=1).swaplevel(axis=1)