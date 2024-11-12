import matplotlib.pyplot as plt
import pandas as pd
from labellines import *
from src.visualization.generate_color_list import *

def plot_vehicle_tracks_clustering_start_end_optics(df_cuid: pd.DataFrame, df_cuid_grouped: pd.DataFrame, clusters: dict, centers: np.ndarray, suptitle: str = 'undefined_suptitle', title: str = 'undefined_title'):
    '''
    Render vehicle tracks for each cluster based on clustering labels and plot cluster centers.

    :param df_cuid: Original dataframe containing 'x' and 'y' coordinates for all vehicle tracks.
    :param df_cuid_grouped: Grouped dataframe containing the vehicle tracks per group.
    :param clusters: Dictionary of lists containing the indices of vehicle tracks belonging to each cluster. 
                     Keys will be interpreted as labels for the groups.
    :param centers: Numpy array containing the (x, y) coordinates of the cluster centers.
    :param suptitle: Supertitle for the plot.
    :param title: Title for the plot.
    '''

    # get axis limits
    min_x = df_cuid['x'].min()
    max_x = df_cuid['x'].max()
    min_y = df_cuid['y'].min()
    max_y = df_cuid['y'].max()

    # Set limits for the plot
    plt.figure(figsize=(10, 10))
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    # create color palette
    colors = generate_color_list(100)

    # List of all track IDs
    track_ids = df_cuid_grouped['track_id'].unique()

    # create list for lines to get a label
    line_list_to_label = []

    # plot tracks
    for i, (cluster_label, cluster_indices) in enumerate(clusters.items()):
        custom_cluster_label = ''
        if f'{cluster_label}' == '-1' or 'outlier' in f'{cluster_label}'.lower():  # Handle outliers
            color = 'gray'
            custom_cluster_label = 'Outlier'
        else:
            color = colors[i]
            custom_cluster_label = f'Cluster #{cluster_label}'

        for index in cluster_indices:  # Iterate over indices in each cluster
            x_vals = df_cuid_grouped['x'].loc[df_cuid_grouped['track_id'] == track_ids[index]].values[0]
            y_vals = df_cuid_grouped['y'].loc[df_cuid_grouped['track_id'] == track_ids[index]].values[0]
            plt.plot(x_vals, y_vals, color=color, linestyle='--', label=custom_cluster_label, linewidth=2, alpha=0.85)
        
        line_list_to_label.append(plt.gca().get_lines()[-1])

    # Plot cluster centers if provided
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], color='red', s=100, label='Cluster Centers', zorder=5)

    # Label lines
    labelLines(line_list_to_label, align=False, fontsize=12, fontweight='bold')

    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys())

    plt.suptitle(f'{suptitle}', fontweight='bold')
    plt.title(title, fontsize=8)
    
    # Plot the figure in the notebook
    plt.show()
    plt.clf()

    print('***All vehicle tracks rendered successfully***')
