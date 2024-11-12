from matplotlib import pyplot as plt
import numpy as np


def get_cluster_legend_handles(colors, labels):

    unique_labels = np.unique(labels)

    # Create a colormap for clusters, ignoring the -1 label (typically noise)
    valid_labels = unique_labels[unique_labels != -1]  # Only non-noise labels

    handles = []
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Noise (-1)'))
    for i, label in enumerate(valid_labels):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10, label=f'Cluster {label}'))

    return handles