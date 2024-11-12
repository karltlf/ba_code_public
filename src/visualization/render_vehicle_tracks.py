import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from pyparsing import col
import re
from labellines import *


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

def render_vehicle_tracks(df_cuid:pd.DataFrame, df_cuid_grouped:pd.DataFrame, intersection_name:str='undefined_intersection', create_slideshow:bool=False, file_path:str='undefined_file_path'):


    min_x = df_cuid['x'].min()
    max_x = df_cuid['x'].max()

    min_y = df_cuid['y'].min()
    max_y = df_cuid['y'].max()

    colors = generate_color_list(30)


    # Set limits for the plot
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    # List of all track IDs
    track_ids = df_cuid_grouped['track_id'].unique()

    path_dir_save_imgs = os.path.expanduser(f'~/Pictures/thesis/{intersection_name}/vehicle_tracks')


    try:
        os.makedirs(path_dir_save_imgs)
        print(f'Created directory {path_dir_save_imgs}')
    except Exception as e:
        print(f'Directory creation went wrong w/ error: {type(e)}')


    if create_slideshow:

        for i in range(len(track_ids)):

            x_vals = df_cuid_grouped['x'].loc[df_cuid_grouped['track_id'] == track_ids[i]].values[0]
            y_vals = df_cuid_grouped['y'].loc[df_cuid_grouped['track_id'] == track_ids[i]].values[0]    
            

            for j in range(i): # plot all prev lines in gray
                x_vals_prev = df_cuid_grouped['x'].loc[df_cuid_grouped['track_id'] == track_ids[j]].values[0]
                y_vals_prev = df_cuid_grouped['y'].loc[df_cuid_grouped['track_id'] == track_ids[j]].values[0]
                plt.plot(x_vals_prev, y_vals_prev, color='gray', linestyle='--')


            plt.plot(x_vals, y_vals, color='red', linestyle='--') # highlight current line

            plt.suptitle(f'#{i} Vehicle path for {intersection_name}', fontweight='bold')

            plt.savefig(os.path.expanduser(path_dir_save_imgs + f'/vehicle_path_{i}.jpg'))
            
            plt.plot(x_vals, y_vals, color='gray', linestyle='--') 


    elif not create_slideshow:

        plt.suptitle(f'All vehicle paths for {intersection_name}', fontweight='bold')


        for i in range(len(track_ids)):

            x_vals = df_cuid_grouped['x'].loc[df_cuid_grouped['track_id'] == track_ids[i]].values[0]
            y_vals = df_cuid_grouped['y'].loc[df_cuid_grouped['track_id'] == track_ids[i]].values[0]    

            plt.plot(x_vals, y_vals, color='gray', linestyle='--')


        plt.savefig(path_dir_save_imgs + f'/all_vehicle_paths.jpg')
        
    # Clear the plot after finishing
    plt.clf()

    print('***All vehicle tracks rendered succesfully***')

def render_vehicle_tracks_highlight_tracks(df_cuid:pd.DataFrame, df_cuid_grouped:pd.DataFrame, index_list:list[str], suptitle:str='undefined_suptitle', title:str='undefined_title', file_name='undefined_file_name', intersection_name='undefined intersection', file_path:str='undefined_file_path'):

    '''
    
    :param file_path: Path consists of /intersection_name/file_path/file_name
    '''


    min_x = df_cuid['x'].min()
    max_x = df_cuid['x'].max()

    min_y = df_cuid['y'].min()
    max_y = df_cuid['y'].max()

    colors = generate_color_list(100)
    print(len(colors))


    # Set limits for the plot
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.figure(figsize=(10, 10), dpi=200)


    # List of all track IDs
    track_ids = df_cuid_grouped['track_id'].unique()

    path_dir_save_imgs = os.path.expanduser(f'~/Pictures/thesis/{intersection_name}/{file_path}')


    try:
        os.makedirs(path_dir_save_imgs)
        print(f'Created directory {path_dir_save_imgs}')
    except Exception as e:
        print(f'Directory creation went wrong w/ error: {type(e)}')



    plt.suptitle(f'{suptitle}', fontweight='bold')
    plt.title(title)


    for i in range(len(track_ids)):

        x_vals = df_cuid_grouped['x'].loc[df_cuid_grouped['track_id'] == track_ids[i]].values[0]
        y_vals = df_cuid_grouped['y'].loc[df_cuid_grouped['track_id'] == track_ids[i]].values[0]    

        if(str(i) in index_list):
            plt.plot(x_vals, y_vals, color=colors[index_list.index(str(i))], linestyle='--', label=f'vehicle index #{i}', zorder=1000-i, linewidth=3)
            plt.legend(loc=2, prop={'size': 6})
        else:
            plt.plot(x_vals, y_vals, color='gray', linestyle='--', alpha=0.5, zorder=0)


    indexes_higlighted = ''
    for index in index_list:
        indexes_higlighted = indexes_higlighted + '_' + index
    
    path = path_dir_save_imgs + f'/{file_name}{indexes_higlighted}.jpg'
    plt.savefig(path)
    
    # Clear the plot after finishing
    plt.clf()

    print('***All vehicle tracks rendered succesfully***')
    print(f'***Saved in {path}***')

def render_single_vehicle_velocities(x:np.array,y:np.array, velocities:np.array, intersection_name:str, file_path:str, file_name:str, limits:dict):

    path = os.path.expanduser(f'~/Pictures/thesis/{intersection_name}/{file_path}')

    fig = plt.figure(1, figsize=(5,5))
    ax  = fig.add_subplot(111)
    pattern = r'\d+'


    # get vehicle_id from file_name
    vehicle_id = 'unknown vehicle id'
    if re.search(pattern,file_name):
        vehicle_id = int(re.search(pattern,file_name).group())
        

    # create path to save fig
    try:
        os.makedirs(path)
        print(f'Created directory {path}')
    except Exception as e:
        print(f'Directory creation went wrong w/ error: {type(e)}') 

    im = plot_multicolor_line(x,y,velocities, limits)
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel('velocity in km/h', rotation=270)
    ax.set_title(f'Velocity in vehicle path for vehicle #{vehicle_id}')

    
    plt.savefig(f'{path}/{file_name}.jpg')
    plt.clf()

def plot_multicolor_line(x:np.array,y:np.array,c:np.array, limits):
    '''
    
    :param limits: needs to contain the keys: v_min,v_max,x_min,x_max,y_min,y_max
    '''
    
    col = cm.jet((c-limits['v_min'])/(limits['v_max']-limits['v_min']))
    ax = plt.gca()
    plt.gcf().set_size_inches(10,10)
    ax.set_xlim(limits['x_min'], limits['x_max'])
    ax.set_ylim(limits['y_min'], limits['y_max']) 
    for i in range(len(x)-1):
        ax.plot([x[i],x[i+1]], [y[i], y[i+1]], c=col[i])

    im = ax.scatter(x, y, c=c, s=0, cmap=cm.jet)
    return im

def render_all_vehicle_velocities(x,y, velocities, intersection_name:str, file_path:str, file_name:str, limits:dict):

    path = os.path.expanduser(f'~/Pictures/thesis/{intersection_name}/{file_path}')

    fig = plt.figure(1, figsize=(5,5))
    ax  = fig.add_subplot(111)
    
 
    

    # create path to save fig
    try:
        os.makedirs(path)
        print(f'Created directory {path}')
    except Exception as e:
        print(f'Directory creation went wrong w/ error: {type(e)}') 

    for i in range(len(x)):
        im = plot_multicolor_line(x[i],y[i],velocities[i], limits)
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel('velocity in km/h', rotation=270)
        
    plt.savefig(f'{path}/{file_name}_all_velocities.jpg')
    plt.clf()

def render_vehicle_tracks_clusters(df_cuid:pd.DataFrame, df_cuid_grouped:pd.DataFrame, clusters:dict, suptitle:str='undefined_suptitle', title:str='undefined_title', file_name='undefined_file_name', intersection_name='undefined intersection', file_path:str='undefined_file_path'):
    '''
    
    :param cluster_groups: Dictionary of lists=clusters containing the indices of vehicle tracks belonging to the relative cluster. Keys will be interpreted as labels for the groups.
    :param file_path: Will `~/Pictures/thesis/intersection_name/file_path/file_name.jpg`
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

    # create dirs to save image
    path_dir_save_imgs = os.path.expanduser(f'~/Pictures/thesis/{intersection_name}/{file_path}')
    try:
        os.makedirs(path_dir_save_imgs)
        print(f'Created directory {path_dir_save_imgs}')
    except Exception as e:
        print(f'Directory creation went wrong w/ error: {type(e)}')

    # create list for lines to get a label
    line_list_to_label =[] # WAY 1
    # line_list_to_label = [None]*1 # WAY 2

    # plot tracks
    plt.suptitle(f'{suptitle}', fontweight='bold')
    plt.title(title, fontsize=8)

    for i, (cluster_label, cluster_indices) in enumerate(clusters.items()):
        if f'{cluster_label}' == '-1' or 'outlier' in f'{cluster_label}'.lower(): # iterate over clusters
            color='gray'
        else:
            color=colors[i]
        
        for index in cluster_indices: # iterate over indices in cluster

            x_vals = df_cuid_grouped['x'].loc[df_cuid_grouped['track_id'] == track_ids[index]].values[0]
            y_vals = df_cuid_grouped['y'].loc[df_cuid_grouped['track_id'] == track_ids[index]].values[0]
            plt.plot(x_vals, y_vals, color=color, linestyle='--', label=f'{cluster_label}', linewidth=2, alpha=0.85)
        
        line_list_to_label.append(plt.gca().get_lines()[-1]) # WAY 1
        # line_list_to_label[0] = plt.gca().get_lines()[-1] # WAY 2
        # labelLines(line_list_to_label,align=True, fontsize=12 ,xvals=x_vals, fontweight='bold') # WAY 2

    labelLines(line_list_to_label,align=False, fontsize=12, fontweight='bold') # WAY 1

    
        

    # save image
    path = path_dir_save_imgs + f'/{file_name}.jpg'
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys())

    plt.savefig(path)
    
    # Clear the plot after finishing
    plt.clf()

    print('***All vehicle tracks rendered succesfully***')
    print(f'***Saved in {path}***')

def render_vehicle_tracks_lat_lon(df_cuid:pd.DataFrame, df_cuid_grouped:pd.DataFrame, intersection_name:str='undefined_intersection', create_slideshow:bool=False, file_path:str='undefined_file_path'):


    min_lat = df_cuid['lat'].min()
    max_lat = df_cuid['lat'].max()

    min_lon = df_cuid['lon'].min()
    max_lon = df_cuid['lon'].max()

    colors = generate_color_list(30)


    # Set limits for the plot
    plt.xlim(min_lat, max_lat)
    plt.ylim(min_lon, max_lon)

    # List of all track IDs
    track_ids = df_cuid_grouped['track_id'].unique()

    path_dir_save_imgs = os.path.expanduser(f'~/Pictures/thesis/{intersection_name}/vehicle_tracks')


    try:
        os.makedirs(path_dir_save_imgs)
        print(f'Created directory {path_dir_save_imgs}')
    except Exception as e:
        print(f'Directory creation went wrong w/ error: {type(e)}')


    if create_slideshow:

        for i in range(len(track_ids)):

            lat_vals = df_cuid_grouped['lat'].loc[df_cuid_grouped['track_id'] == track_ids[i]].values[0]
            lon_vals = df_cuid_grouped['lon'].loc[df_cuid_grouped['track_id'] == track_ids[i]].values[0]    
            

            for j in range(i): # plot all prev lines in gray
                lat_vals_prev = df_cuid_grouped['lat'].loc[df_cuid_grouped['track_id'] == track_ids[j]].values[0]
                lon_vals_prev = df_cuid_grouped['lon'].loc[df_cuid_grouped['track_id'] == track_ids[j]].values[0]
                plt.plot(lat_vals_prev, lon_vals_prev, color='gray', linestyle='--')


            plt.plot(lat_vals, lon_vals, color='red', linestyle='--') # highlight current line

            plt.suptitle(f'#{i} Vehicle path for {intersection_name}', fontweight='bold')

            plt.savefig(os.path.expanduser(path_dir_save_imgs + f'/vehicle_path_{i}.jpg'))
            
            plt.plot(lat_vals, lon_vals, color='gray', linestyle='--') 


    elif not create_slideshow:

        plt.suptitle(f'All vehicle paths for {intersection_name}', fontweight='bold')


        for i in range(len(track_ids)):

            lat_vals = df_cuid_grouped['lat'].loc[df_cuid_grouped['track_id'] == track_ids[i]].values[0]
            lon_vals = df_cuid_grouped['lon'].loc[df_cuid_grouped['track_id'] == track_ids[i]].values[0]    

            plt.plot(lat_vals, lon_vals, color='gray', linestyle='--')


        plt.savefig(path_dir_save_imgs + f'/all_vehicle_paths.jpg')
        
    # Clear the plot after finishing
    plt.clf()

    print('***All vehicle tracks rendered succesfully***')

def render_vehicle_tracks_clusters_lat_lon(df_cuid:pd.DataFrame, df_cuid_grouped:pd.DataFrame, clusters:dict, suptitle:str='undefined_suptitle', title:str='undefined_title', file_name='undefined_file_name', intersection_name='undefined intersection', file_path:str='undefined_file_path'):
    '''
    
    :param cluster_groups: Dictionary of lists=clusters containing the indices of vehicle tracks belonging to the relative cluster. Keys will be interpreted as labels for the groups.
    :param file_path: Will `~/Pictures/thesis/intersection_name/file_path/file_name.jpg`
    '''

    # get axis limits
    min_lat = df_cuid['lat'].min()
    max_lat = df_cuid['lat'].max()
    min_lon = df_cuid['lon'].min()
    max_lon = df_cuid['lon'].max()

    # Set limits for the plot
    plt.xlim(min_lat, max_lat)
    plt.ylim(min_lon, max_lon)
    plt.figure(figsize=(10, 10), dpi=200)

    # create color palette
    colors = generate_color_list(100)

    # List of all track IDs
    track_ids = df_cuid_grouped['track_id'].unique()

    # create dirs to save image
    path_dir_save_imgs = os.path.expanduser(f'~/Pictures/thesis/{intersection_name}/{file_path}')
    try:
        os.makedirs(path_dir_save_imgs)
        print(f'Created directory {path_dir_save_imgs}')
    except Exception as e:
        print(f'Directory creation went wrong w/ error: {type(e)}')

    # create list for lines to get a label
    line_list_to_label =[] # WAY 1
    # line_list_to_label = [None]*1 # WAY 2

    # plot tracks
    plt.suptitle(f'{suptitle}', fontweight='bold')
    plt.title(title, fontsize=8)

    for i, (cluster_label, cluster_indices) in enumerate(clusters.items()):
        if f'{cluster_label}' == '-1' or 'outlier' in f'{cluster_label}'.lower(): # iterate over clusters
            color='gray'
        else:
            color=colors[i]
        
        for index in cluster_indices: # iterate over indices in cluster

            lat_vals = df_cuid_grouped['lat'].loc[df_cuid_grouped['track_id'] == track_ids[index]].values[0]
            lon_vals = df_cuid_grouped['lon'].loc[df_cuid_grouped['track_id'] == track_ids[index]].values[0]
            plt.plot(lat_vals, lon_vals, color=color, linestyle='--', label=f'{cluster_label}', linewidth=2, alpha=0.85)
        
        line_list_to_label.append(plt.gca().get_lines()[-1]) # WAY 1
        # line_list_to_label[0] = plt.gca().get_lines()[-1] # WAY 2
        # labelLines(line_list_to_label,align=True, fontsize=12 ,xvals=x_vals, fontweight='bold') # WAY 2

    labelLines(line_list_to_label,align=False, fontsize=12, fontweight='bold') # WAY 1


    # save image
    path = path_dir_save_imgs + f'/{file_name}.jpg'
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys())

    plt.savefig(path)
    
    # Clear the plot after finishing
    plt.clf()

    print('***All vehicle tracks rendered succesfully***')
    print(f'***Saved in {path}***')
