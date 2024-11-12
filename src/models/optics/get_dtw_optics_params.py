def get_dtw_optics_params(optics_params, metric):
    optics_params = optics_params[metric]
    dtw_dist_matrix_key = optics_params['dtw_key']
    max_eps = optics_params['epsilon']
    min_samples = optics_params['min_samples']
    cluster_method = optics_params['cluster_method']
    xi = optics_params['xi']
    kwargs = {
        'max_eps': max_eps,
        'min_samples': min_samples,
        'cluster_method': cluster_method,
        'xi': xi
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return dtw_dist_matrix_key, filtered_kwargs