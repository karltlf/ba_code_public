import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_vehicle_tracks_in_notebook(ax, df_cuid:pd.DataFrame, df_cuid_grouped:pd.DataFrame, title:str='', **kwargs):
    '''
    
    :param df_cuid: Used for evaluating the x and y limits in plotting
    '''


    min_x = df_cuid['x'].min()
    max_x = df_cuid['x'].max()

    min_y = df_cuid['y'].min()
    max_y = df_cuid['y'].max()

    # Set limits for the plot
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # List of all track IDs
    track_ids = df_cuid_grouped['track_id'].unique()

    ax.set_title(f'{title}')


    for i in range(len(track_ids)):

        x_vals = df_cuid_grouped['x'].loc[df_cuid_grouped['track_id'] == track_ids[i]].values[0]
        y_vals = df_cuid_grouped['y'].loc[df_cuid_grouped['track_id'] == track_ids[i]].values[0]    

        ax.plot(x_vals, y_vals, **kwargs)

        
    # Clear the plot after finishing

    print('***All vehicle tracks rendered succesfully***')