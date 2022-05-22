import pandas as pd
import numpy as np

__all__ = [
    'pd_add_colwise_percentage',
    'pd_column_locs',
    'pd_dataset_split',
    'pd_dataset_label_proportion',
]

def pd_add_colwise_percentage(df):
    """ Add column-wise percentage to each cell. """
    df = df.replace(to_replace=[np.nan], value=0, inplace=False)
    percentage_df = df.div(df.sum(axis=0), axis=1)

    def add_percent(v, p):
        return f'{int(v)} ({p*100:.1f}%)'
    return pd.DataFrame(np.vectorize(add_percent)(df, percentage_df),
                        columns=df.columns,
                        index=df.index)


def pd_column_locs(df, cols):
    return [df.columns.get_loc(x) for x in cols]


def pd_dataset_split(df, cols, keys=['train', 'validate', 'test']):
    import collections
    def fix_ordering(d): return \
        collections.OrderedDict([(k, d[k]) for k in keys])
    m = {'Total': fix_ordering(df['split'].value_counts().to_dict())}
    for k in cols:
        m[k] = fix_ordering(df[df[k].notnull()]
                            ['split'].value_counts().to_dict())
    stat_df = pd.DataFrame(m)
    stat_df = pd_add_colwise_percentage(stat_df)
    return stat_df


def pd_dataset_label_proportion(df, cols):
    m = {}
    for k in cols:
        m[k] = df[df[k].notnull()][k].value_counts().to_dict()
    stat_df = pd.DataFrame(m)
    stat_df = pd_add_colwise_percentage(stat_df)
    return stat_df
