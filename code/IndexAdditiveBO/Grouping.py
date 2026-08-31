import numpy as np
import math
from sklearn.cluster import KMeans, AgglomerativeClustering
import torch
import networkx as nx
import community as community_louvain
from IndexAdditiveBO.interaction_metric import AverageImprovementInteraction,MarginalLikelihoodInteraction,RegressionInteractionScore


def group_sampler(grouping_strategy, E, n_groups = 1, max_group_size = 5):
    if grouping_strategy == "Kmeans":
        return KMeansDistanceMatrix(E, n_groups)
    elif grouping_strategy == "CapacitatedKmeans":
        return CapacitatedKMeansDistanceMatrix(E, n_groups, max_capacity=max_group_size)
    elif grouping_strategy == "Random":
        return RandomGroupingSampler(E, n_groups)
    elif grouping_strategy == "Louvain":
        return LouvainGroupingSampler(E)
    elif grouping_strategy == "Hierarchical":
        return HierarchicalGroupingSampler(E, threshold=0.5)
    elif grouping_strategy == "FCM":
        return FCMGroupingSampler(E, cluster_size_upper=max_group_size)
    else:
        raise ValueError(f"Unknown grouping strategy: {grouping_strategy}")

def FCMGroupingSampler(E, cluster_size_upper,n_clusters=8, m=2.0, max_iter=200, tol=1e-5):
    D = 1 - E
    d = D.shape[0]
    X = D.astype(np.float64)
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    U = np.random.rand(n_clusters, d)
    U = U / np.sum(U, axis=0, keepdims=True)

    for _ in range(max_iter):
        U_old = U.copy()
        Um = U ** m
        centers = (Um @ X) / (Um.sum(axis=1, keepdims=True) + 1e-8)

        dist = np.zeros((n_clusters, d))
        for c in range(n_clusters):
            diff = X - centers[c]
            dist[c] = np.sum(diff * diff, axis=1) + 1e-8

        power = 1.0 / (m - 1.0)
        tmp = dist ** (-power)
        U = tmp / np.sum(tmp, axis=0, keepdims=True)

        if np.linalg.norm(U - U_old) < tol:
            break
    groups = [[] for _ in range(n_clusters)]
    cluster_sizes = np.zeros(n_clusters, dtype=int)
    for i in range(d):
        sorted_clusters = np.argsort(U[:, i])[::-1]
        for c in sorted_clusters:
            if cluster_size_upper is None or cluster_sizes[c] < cluster_size_upper:
                groups[c].append(i)
                cluster_sizes[c] += 1
                break

    for c in range(n_clusters):
        remaining_vars = set(range(d)) - set(groups[c])
        candidates = sorted(remaining_vars, key=lambda i: U[c, i], reverse=True)
        for var in candidates:
            if cluster_size_upper is not None and cluster_sizes[c] >= cluster_size_upper:
                break
            groups[c].append(var)
            cluster_sizes[c] += 1

    unique_groups = []
    seen = set()
    for g in groups:
        g_sorted = tuple(sorted(g))
        if g_sorted not in seen:
            seen.add(g_sorted)
            unique_groups.append(list(g_sorted))

    return unique_groups

def KMeansDistanceMatrix(E, n_clusters, max_iter=100):
    D = 1 - E
    n = D.shape[0]
    centers = np.random.choice(n, size=n_clusters, replace=False)
    labels = np.zeros(n, dtype=int)
    for it in range(max_iter):
        for i in range(n):
            dist_to_centers = D[i, centers]
            labels[i] = np.argmin(dist_to_centers)
        new_centers = []
        for k in range(n_clusters):
            cluster_nodes = np.where(labels == k)[0]
            if len(cluster_nodes) == 0:
                new_centers.append(np.random.choice(n))
                continue
            sub_D = D[np.ix_(cluster_nodes, cluster_nodes)]
            avg_dist = sub_D.mean(axis=1)
            new_centers.append(cluster_nodes[np.argmin(avg_dist)])
        new_centers = np.array(new_centers)
        if np.all(new_centers == centers):
            break
        centers = new_centers
    groups = [[] for _ in range(n_clusters)]
    for i, lbl in enumerate(labels):
        groups[lbl].append(i)
    return groups


def CapacitatedKMeansDistanceMatrix(E, n_clusters, max_capacity, max_iter=80):
    D = 1 - E
    n = D.shape[0]
    centers = np.random.choice(n, size=n_clusters, replace=False)
    labels = np.zeros(n, dtype=int)
    for it in range(max_iter):
        cluster_sizes = np.zeros(n_clusters, dtype=int)
        for i in range(n):
            dist_to_centers = D[i, centers]
            sorted_idx = np.argsort(dist_to_centers)
            for k in sorted_idx:
                if cluster_sizes[k] < max_capacity:
                    labels[i] = k
                    cluster_sizes[k] += 1
                    break
        new_centers = []
        for k in range(n_clusters):
            cluster_nodes = np.where(labels == k)[0]
            if len(cluster_nodes) == 0:
                new_centers.append(np.random.choice(n))
                continue
            sub_D = D[np.ix_(cluster_nodes, cluster_nodes)]
            avg_dist = sub_D.mean(axis=1)
            new_centers.append(cluster_nodes[np.argmin(avg_dist)])
        new_centers = np.array(new_centers)

        if np.all(new_centers == centers):
            break
        centers = new_centers

    groups = [[] for _ in range(n_clusters)]
    for i, lbl in enumerate(labels):
        groups[lbl].append(i)
    return groups

def RandomGroupingSampler(matrix, n_groups):
    d = matrix.shape[0]
    perm = np.random.permutation(d)
    groups = np.array_split(perm, n_groups)
    return [list(g) for g in groups]

def LouvainGroupingSampler(E):
    """
    Louvain
    """
    G = nx.Graph()
    d = E.shape[0]
    for i in range(d):
        for j in range(i+1, d):
            if E[i, j] > 0:
                G.add_edge(i, j, weight=E[i, j])
    partition = community_louvain.best_partition(G, resolution=1.5, randomize=None)

    comms = {}
    for n, cid in partition.items():
        comms.setdefault(cid, []).append(n)
    return list(comms.values())

def HierarchicalGroupingSampler(E, threshold=0.5, linkage='average'):
    D = 1 - E
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        affinity='precomputed',
        linkage=linkage
    )
    labels = clustering.fit_predict(D)
    groups = {}
    for i, lbl in enumerate(labels):
        groups.setdefault(lbl, []).append(i)
    return list(groups.values())
