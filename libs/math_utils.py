import numpy as np
from math import radians, sin, cos, sqrt, asin, fabs
from numpy.linalg import norm
import collections
from sklearn.cluster import KMeans, DBSCAN

def cosSim(v1: "ndarray", v2: "ndarray") -> "ndarray":
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

def standardization(x, axis=None, ddof=0):
    alpha = 1e-10
    x_mean = x.mean(axis=axis, keepdims=True)
    x_std = x.std(axis=axis, keepdims=True, ddof=ddof)

    return (x - x_mean) / (x_std + alpha)

def normalization(x, amin=0, amax=1):
    xmax = x.max()
    xmin = x.min()
    if xmin == xmax:
        return np.ones_like(x)
    return (amax - amin) * (x - xmin) / (xmax - xmin) + amin

def gini(y):
    m = statistics.mean(y)
    n = len(y)
    a = 2 * m * (n * (n - 1))
    ysum = 0
    for i in range(n):
        for j in range(n):
            ysum = ysum + (fabs(y[i] - y[j]))
    return(ysum / a)

def calc_distance_and_neighbor_point(a, b, p) -> "tuple":
    """
    Calculate the nearest point and distance from point 'p' to the line segment 'ab'.
    
    Parameters:
    a, b (ndarray): Endpoints of the line segment.
    p (ndarray): Point from which the distance is measured.
    
    Returns:
    tuple: Nearest point on the line segment and the distance to 'p'.
    """
    ap, ab, bp = p - a, b - a, p - b
    if np.dot(ap, ab) < 0:
        return a, norm(ap)
    elif np.dot(bp, ab) > 0:
        return b, norm(bp)
    else:
        proj_length = np.dot(ap, ab) / norm(ab)
        nearest_point = a + ab / norm(ab) * proj_length
        return nearest_point, norm(p - nearest_point)
    
def assign_to_nearest_anchor(vectors, anchors):
    """
    Assign each vector to the nearest anchor.
    :param vectors: Numpy array of vectors.
    :param anchors: Numpy array of anchor points.
    :return: Tuple containing the indices of the nearest anchors and the count of elements per cluster.
    """
    distances = np.linalg.norm(vectors[:, np.newaxis] - anchors, axis=2)
    nearest_centroid_indices = np.argmin(distances, axis=1)
    element_num_in_cluster = collections.Counter(nearest_centroid_indices)
    return nearest_centroid_indices, element_num_in_cluster  
  
def do_kmeans(cluster_num, vecs, with_center=False, seed=20210401, weight=None):
    """
    Perform K-means clustering on the provided vectors.
    :param cluster_num: Number of clusters.
    :param vecs: Numpy array of vectors to cluster.
    :param with_center: Whether to return the cluster centers.
    :param seed: Random state seed for reproducibility.
    :param weight: Sample weights.
    :return: Cluster labels, count of elements per cluster, and optionally cluster centers.
    """
    kmeans = KMeans(n_clusters=cluster_num, random_state=seed).fit(vecs, sample_weight=weight)
    result = kmeans.labels_
    element_num_in_cluster = collections.Counter(result)
    return (list(result), element_num_in_cluster, kmeans.cluster_centers_) if with_center else (list(result), element_num_in_cluster)

def do_DBSCAN(vec, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering on the provided vectors.
    :param vec: Numpy array of vectors to cluster.
    :param eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    :return: Cluster labels and the number of clusters.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(vec)
    labels = dbscan.labels_
    cluster_num = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, cluster_num