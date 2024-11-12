import pandas as pd

# Create tuples of coordinate vectors with adjustable feature count
def get_tuple_list(df_cuid_grouped: pd.DataFrame, features: list[str]) -> list:
    '''
    Returns a list of tuples for each entry in a grouped dataframe. The tuple can have two or more elements depending on the number of features provided.

    :param df_cuid_grouped: Grouped dataframe
    :param features: List of features as strings, supported strings: 'x', 'y', 'vx', 'vy', 'z', etc.
    :return: A list of lists, where each inner list contains tuples of the feature values for each track_id
    '''
    list_all_tuples = []
    list_individual_tuples = []

    # Iterate through each entry in the grouped dataframe
    for i in range(len(df_cuid_grouped[features[0]])):  # Assuming all features have the same length per track
        for j in range(len(df_cuid_grouped[features[0]][i])):  # Iterate over all measurements for one track_id
            # Create a tuple of feature values for each measurement point
            list_item = tuple(df_cuid_grouped[feature][i][j] for feature in features)
            list_individual_tuples.append(list_item)  # Add tuple to list for individual track_id
        list_all_tuples.append(list_individual_tuples)  # Add tuple list to list for all track_ids
        list_individual_tuples = []  # Reset list for the next track_id

    return list_all_tuples
