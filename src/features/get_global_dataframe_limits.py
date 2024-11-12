import pandas as pd

def get_global_dataframe_limits(df_cuid:pd.DataFrame, feature_list:list[str]) -> dict:

    '''
    Returns the min and max values for the specified dataframe and its features.

    :param df_cuid: The dataframe to get the limits from.
    :param feature_list: The list of features (as `str`) to get the limits from.
    :return limits: A dictionary with the min and max values for each feature, `feature_min` and `feature_max`.
    '''

    limits = {}
    for feature in feature_list:
        limits[f'{feature}_min'] = df_cuid[feature].min()
        limits[f'{feature}_max'] = df_cuid[feature].max()
    return limits

