import pandas as pd
import numpy as np

def pd_add_colwise_percentage(df):
    """ Add column-wise percentage to each cell. """
    df = df.replace(to_replace=[np.nan], value=0, inplace=False)
    percentage_df = df.div(df.sum(axis=0), axis=1)
    def add_percent(v, p):
        return f'{int(v)} ({p*100:.0f}%)'
    return pd.DataFrame(np.vectorize(add_percent)(df, percentage_df),
                        columns=df.columns,
                        index=df.index)