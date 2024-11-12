from matplotlib import pyplot as plt
from src.visualization.generate_color_list import *
from labellines import *
import pandas as pd

def render_vehicle_track_cluster_in_notebook(axis: plt.axes, df_cuid: pd.DataFrame, df_cuid_grouped: pd.DataFrame, clusters: dict):

    # Get axis limits
    min_x = df_cuid['x'].min()
    max_x = df_cuid['x'].max()
    min_y = df_cuid['y'].min()
    max_y = df_cuid['y'].max()

    # Set limits for the plot
    axis.set_xlim(min_x, max_x)
    axis.set_ylim(min_y, max_y)
    
    # Create color palette
    colors = generate_color_list(100)

    # List of all track IDs
    track_ids = df_cuid_grouped['track_id'].unique()

    # Create list for lines to get a label
    line_list_to_label = []

    # Plot tracks
    for i, (cluster_label, cluster_indices) in enumerate(clusters.items()):
        custom_cluster_label = ''
        if f'{cluster_label}' == '-1' or 'outlier' in f'{cluster_label}'.lower():  # Check if label is -1 or an outlier
            color = 'gray'
            custom_cluster_label = 'Outlier'
        else:
            color = colors[i]
            custom_cluster_label = f'Cluster #{cluster_label}'
        
        for index in cluster_indices:  # Iterate over indices in cluster
            x_vals = df_cuid_grouped['x'].loc[df_cuid_grouped['track_id'] == track_ids[index]].values[0]
            y_vals = df_cuid_grouped['y'].loc[df_cuid_grouped['track_id'] == track_ids[index]].values[0]
            axis.plot(
                x_vals, y_vals,
                color=color,
                linestyle='--',
                label=custom_cluster_label,
                linewidth=2,
                alpha=0.85,
                zorder=1000
            )

            # Add arrows only for clusters with a valid label (not -1)
            if cluster_label != -1:
                num_points = len(x_vals)
                interval = max(1, num_points // 10)  # Adjust the divisor to change the number of arrows

                for idx in range(0, num_points - 1, interval * 2):  # Adjust interval for half as frequent arrows
                    start_x = x_vals[idx]
                    start_y = y_vals[idx]
                    end_x = x_vals[idx + 1]
                    end_y = y_vals[idx + 1]
                    axis.annotate(
                        '',
                        xy=(end_x, end_y),
                        xytext=(start_x, start_y),
                        arrowprops=dict(
                            arrowstyle='->',
                            color=color,
                            alpha=0.85,
                            lw=2,               # Increase line width
                            mutation_scale=20   # Increase arrow size
                        )
                    )

        line_list_to_label.append(axis.get_lines()[-1])

    labelLines(line_list_to_label, align=False, fontsize=16, fontweight='bold', zorder=10000)

    handles, labels = axis.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    legend = axis.legend(unique_labels.values(), unique_labels.keys())
    legend.set_zorder(1000000)
    print('***All vehicle tracks rendered successfully***')
