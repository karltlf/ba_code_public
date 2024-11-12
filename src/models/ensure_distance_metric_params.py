import numpy as np

def ensure_distance_metric_params(X, metrics:list[str], **kwargs):
    """
    Ensures that the necessary parameters for each metric are passed.
    IF ONLY ONE METRIC PASSED: PASS AS LIST AND UNPACK RETURN DICT WITH metric_params[distance_metric]

    Parameters:
    - X: np.ndarray, the dataset
    - metrics: list of str, the metrics to be used
    - kwargs: additional parameters (e.g., p for Minkowski)

    Returns:
    - metric_params_dict: dict, necessary parameters for the metric (if any)
    """
    
    metric_params_dict = {}
    
    for metric in metrics:
        if metric == 'seuclidean':
            # seuclidean requires variance (V) for each feature
            metric_params_dict[metric] = {'V': np.var(X, axis=0)}
        
        elif metric == 'mahalanobis':
            # mahalanobis requires the inverse covariance matrix
            VI = np.linalg.inv(np.cov(X.T))
            metric_params_dict[metric] = {'VI': VI}
        
        elif metric == 'minkowski':
            # minkowski requires a p parameter (set to a default value of 2)
            metric_params_dict[metric] = {'p': kwargs.get('p', 2)}
        
        else:
            # For other metrics, no additional parameters are required
            metric_params_dict[metric] = None

    return metric_params_dict
