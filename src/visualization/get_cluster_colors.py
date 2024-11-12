import numpy as np
import matplotlib.pyplot as plt

def get_cluster_colors(labels):
    """
    Assign unique, vibrant colors to each unique label, using gray for -1 and non-gray, striking colors otherwise.
    
    :param labels: array-like, shape (n_samples,)
    :return: point_colors (list of colors for each point), unique_colors (list of unique colors assigned)
    """
    unique_labels = np.unique(labels)
    
    # Use a more highlighting color map such as 'tab20c' for vibrant and bold colors
    color_palette = plt.get_cmap('tab20c')
    
    # Assign gray to the noise label (-1)
    label_color_mapping = {}
    unique_colors = []
    color_idx = 0  # To cycle through colors

    for label in unique_labels:
        if label == -1:
            label_color_mapping[label] = 'gray'  # Gray for label -1
        else:
            color = color_palette(color_idx / len(unique_labels))  # Assign a color from the palette
            label_color_mapping[label] = color
            unique_colors.append(color)
            color_idx += 1
    
    # Create the list of colors for each point based on the labels
    point_colors = [label_color_mapping[label] for label in labels]
    
    return point_colors, unique_colors
