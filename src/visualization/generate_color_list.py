from matplotlib import pyplot as plt
import numpy as np


def generate_color_list(num_colors):
    """
    Generate a list of distinct colors for plotting.
    
    :param num_colors: Number of distinct colors needed.
    :return: List of RGB color codes.
    """
    # Start with 'tab20', 'tab20b', and 'tab20c' color maps
    color_maps = ['tab20', 'tab20b', 'tab20c']
    
    # Combine the colors from the available color maps
    all_colors = []
    for cmap_name in color_maps:
        cmap = plt.get_cmap(cmap_name)
        all_colors.extend(cmap.colors)
    
    # If more colors are needed, generate from 'hsv' colormap
    if num_colors > len(all_colors):
        # Generate additional colors using the 'hsv' colormap
        extra_colors_needed = num_colors - len(all_colors)
        hsv_colors = plt.get_cmap('hsv')(np.linspace(0, 1, extra_colors_needed))
        all_colors.extend(hsv_colors)
    
    return all_colors[:num_colors]