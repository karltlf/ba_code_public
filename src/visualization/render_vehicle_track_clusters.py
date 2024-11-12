import os
from matplotlib import pyplot as plt
from src.visualization.generate_color_list import *
from labellines import *
import pandas as pd
from IPython.display import display, clear_output


def render_vehicle_track_clusters(df_cuid:pd.DataFrame, df_cuid_grouped:pd.DataFrame, clusters:dict, suptitle:str='undefined_suptitle', title:str='undefined_title'):
    '''
    
    :param cluster_groups: Dictionary of lists=clusters containing the indices of vehicle tracks belonging to the relative cluster. Keys will be interpreted as labels for the groups.
    :param file_path: Will default to `~/Documents/Uni/10-SS-24/Bachelorarbeit/ba_code_project/reports/figures/file_name.jpg`, otherwise `file_path` is used
    '''

    # get axis limits
    min_x = df_cuid['x'].min()
    max_x = df_cuid['x'].max()
    min_y = df_cuid['y'].min()
    max_y = df_cuid['y'].max()

    # Set limits for the plot
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.figure(figsize=(10, 10), dpi=200)

    # create color palette
    colors = generate_color_list(100)

    # List of all track IDs
    track_ids = df_cuid_grouped['track_id'].unique()

    # create list for lines to get a label
    line_list_to_label =[]

    # plot tracks
    

    for i, (cluster_label, cluster_indices) in enumerate(clusters.items()):
        custom_cluster_label = ''
        if f'{cluster_label}' == '-1' or 'outlier' in f'{cluster_label}'.lower(): # iterate over clusters
            color='gray'
            custom_cluster_label = 'Outlier'
        else:
            color=colors[i]
            custom_cluster_label=f'Cluster #{cluster_label}'
        
        for index in cluster_indices: # iterate over indices in cluster

            x_vals = df_cuid_grouped['x'].loc[df_cuid_grouped['track_id'] == track_ids[index]].values[0]
            y_vals = df_cuid_grouped['y'].loc[df_cuid_grouped['track_id'] == track_ids[index]].values[0]
            plt.plot(x_vals, y_vals, color=color, linestyle='--', label=custom_cluster_label, linewidth=2, alpha=0.85)
        
        line_list_to_label.append(plt.gca().get_lines()[-1])

    labelLines(line_list_to_label,align=False, fontsize=12, fontweight='bold')

    
        

    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.suptitle(f'{suptitle}', fontweight='bold')
    plt.title(title, fontsize=8)
    plt.legend(unique_labels.values(), unique_labels.keys())
    plt.show()
    plt.clf()

    print('***All vehicle tracks rendered succesfully***')