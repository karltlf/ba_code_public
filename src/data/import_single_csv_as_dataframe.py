import pandas as pd
import ast
from typing import Tuple
from importable_functions.get_velocities import *

def import_single_csv_as_workable_dataframe(path:str, group_by:str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Imports csvs as dataframes with `x,` `y,` `vx`, `vy` as number objects
    
    :return df_cuid: csv as pandas DataFrame with `x,` `y,` `vx`, `vy` as number objects
    :return df_cuid: csv as pandas DataFrame grouped by `track_id` with `x,` `y,` `vx`, `vy` as number objects
    '''
    df_cuid = pd.read_csv(path, index_col=0)
    df_cuid = df_cuid.loc[df_cuid['agent_type'].isin(['Truck', 'Car', 'Bike', 'truck','car','bike'])]

    # df_cuid['x'] = df_cuid['x'].apply(lambda x: ast.literal_eval(x))
    # df_cuid['y'] = df_cuid['y'].apply(lambda y: ast.literal_eval(y))
    # df_cuid['vx'] = df_cuid['vx'].apply(lambda vx: ast.literal_eval(vx))
    # df_cuid['vy'] = df_cuid['vy'].apply(lambda vy: ast.literal_eval(vy))

    df_cuid_grouped = df_cuid.groupby(group_by, as_index=False).agg(list)


    return (df_cuid,df_cuid_grouped)