def get_subcluster_paths_from_subcluster_indices(subcluster_indices, df_cuid_grouped):

    # Extract subcluster paths from the DataFrame
    subcluster_paths = {}

    # iterate over all clusters
    for i, (cluster_key, subclusters) in enumerate(subcluster_indices.items()):
        
        # create a dict for each cluster
        subcluster_paths[cluster_key] = {}

        # iterate over all subclusters in cluster
        for j, (subcluster_key, indices) in enumerate(subclusters.items()):
            # create a dict for each subcluster
            subcluster_paths[cluster_key][subcluster_key] = {}
            # iterate over all indices in subcluster
            for i, idx in enumerate(indices):
                # save indices in subcluster_paths under cluster_key and subcluster_key and the index within the subcluster
                x, y = df_cuid_grouped.iloc[idx][['x','y']]
                subcluster_paths[cluster_key][subcluster_key][i] = [ # list of lists of x and y coordinates
                    x,
                    y
                ]

    return subcluster_paths