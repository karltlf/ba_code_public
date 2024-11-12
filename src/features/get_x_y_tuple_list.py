import pandas as pd

# create x,y coordinate vectors
def get_x_y_tuple_list(df_cuid_grouped:pd.DataFrame, x_y_features: list[str]) -> list:
    '''
    Returns a list of [x,y] element lists for each entry in a grouped dataframe

    :param df_cuid_grouped: Grouped dataframe
    :param x_y_features: List of two features as strings, supported strings: 'x','y','vx','vy'
    '''
    list_all_tuples = []
    list_individual_tuples = []
    x_feature = x_y_features[0]
    y_feature = x_y_features[1]
    for i in range(len(df_cuid_grouped[x_feature])):
        for j in range(len(df_cuid_grouped[x_feature][i])): # iterate over all measurements for one track_id
            list_item = [df_cuid_grouped[x_feature][i][j], df_cuid_grouped[y_feature][i][j]] # create x-y tuples for each measumerement point
            list_individual_tuples.append(list_item) # add tuple to list for individual track_id
        list_all_tuples.append(list_individual_tuples) # add tuple list to list for all track_ids
        list_individual_tuples = []

    return list_all_tuples