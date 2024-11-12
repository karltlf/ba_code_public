from sklearn.cluster import OPTICS
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np

def calculate_sse(X, labels, cluster_centers):
    sse = 0
    for i, center in enumerate(cluster_centers):
        cluster_points = X[labels == i]
        sse += np.sum((cluster_points - center) ** 2)
    return sse

def calculate_log_likelihood(sse, n):
    variance = sse / n  # Variance estimate
    log_likelihood = -n * np.log(variance)
    return log_likelihood

def calculate_aic_bic_for_optics(X, labels, cluster_centers):
    n = len(X)
    k = len(cluster_centers)  # Number of clusters
    
    # Calculate SSE (within-cluster sum of squared errors)
    sse = calculate_sse(X, labels, cluster_centers)
    
    # Calculate log-likelihood
    log_likelihood = calculate_log_likelihood(sse, n)
    
    # Calculate AIC and BIC
    aic = 2 * k - 2 * log_likelihood
    bic = np.log(n) * k - 2 * log_likelihood
    
    return aic, bic
