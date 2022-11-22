import pandas as pd
import numpy as np

__all__ = [
    "pd_add_colwise_percentage",
    "pd_column_locs",
    "pd_dataset_split",
    "pd_dataset_label_proportion",
    "pd_apply_fn_to_list_flattened",
    "pd_sort_rows_by_avg_ranking",
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
    colsum = stat_df.sum(axis=0).apply(lambda x: int(x))
    stat_df = pd_add_colwise_percentage(stat_df)
    stat_df.loc['total', :] = colsum
    return stat_df


def pd_dataset_label_proportion(df, cols):
    """ Creates a dataframe where 
            rows := possible labels
            cols := column names
        and the cell contain the counts of the label 
            occuring under that column name
    """
    m = {}
    for k in cols:
        m[k] = df[df[k].notnull()][k].value_counts().to_dict()
    stat_df = pd.DataFrame(m)
    colsum = stat_df.sum(axis=0).apply(lambda x: int(x))
    stat_df = pd_add_colwise_percentage(stat_df)
    stat_df.loc['total', :] = colsum
    return stat_df


def pd_apply_fn_to_list_flattened(fn, df, by, col):
    """Use explode to flatten `df[col]` where cell value
        are List<object>. Then apply `fn` to generate an 
        updated list of values. Then compressed to list form.
            
            fn: List<str> -> List<object>

        ```
        col = 'sents'
        s = pd_apply_fn_to_list_flattened(
            lambda x: x, df, 'study_id', col)
        assert(s.equals(df[col]))
        ```

        Note explode convert `[]` -> `np.nan`
        If `fn` requires non-null vaules, need to make sure 
            there is no `[]` in `df[col]`.
    """
    # Pick rows with non-nan value.
    notnull_indices = df.index[df[col].notnull()]
    # Transform each element of a list-like to row.
    dfe = df.loc[notnull_indices].explode(col)

    dfe[col] = fn(dfe[col])
    # Transform to list-like value.
    # re-order `col` by ordering of `df[by]`
    dfe = (dfe.groupby(by)
              .agg({col: lambda x: x.to_list()})
              .reindex(index=df[by])
              .reset_index())
    return dfe[col]


def pd_sort_rows_by_avg_ranking(df, metrics_lower_the_better=tuple(), drop_ranking=False):
    """Sort rows of dataframe `df` by average of nuermical columns. """
    cols = df.select_dtypes([np.number]).columns
    def is_ascending(x):
        # ascending=True if metrics lower the better
        if isinstance(x,  tuple):
            x = x[-1]
        return x.startswith(tuple(metrics_lower_the_better))
    ascendings = list(map(is_ascending, cols))
    rankings = [df[col].rank(ascending=ascending).to_numpy() 
                for col, ascending in zip(cols, ascendings)]
    df = df.copy()
    df.loc[:,'ranking'] = np.array(rankings).mean(0)
    df = df.sort_values('ranking', ascending=True)
    if drop_ranking:
        df = df.drop(columns=['ranking'])
    return df