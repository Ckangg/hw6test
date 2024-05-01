"""
Work with Jarvis-Patrick clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle
from scipy.special import comb

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################

def compute_distances(data):
    # Compute pairwise Euclidean distances between points
    distances = np.linalg.norm(data[:, np.newaxis, :] - data, axis=2)
    return distances

def compute_sse(data, clusters):
    centroids = np.array([np.mean(data[clusters == i], axis=0) for i in np.unique(clusters)])
    # Adjust cluster labels to start from 0
    clusters_adjusted = clusters - 1
    sse = np.sum((data - centroids[clusters_adjusted])**2)
    return sse
def calculate_ari(true_labels,predicted_labels):
    contingency_matrix = np.histogram2d(true_labels, predicted_labels, bins=(np.unique(true_labels).size, np.unique(predicted_labels).size))[0]

    # Sum the combinatorics for each row and column
    sum_comb_c = sum(comb(n_c, 2) for n_c in np.sum(contingency_matrix, axis=1))
    sum_comb_k = sum(comb(n_k, 2) for n_k in np.sum(contingency_matrix, axis=0))

    # Sum the combinatorics for the whole matrix
    sum_comb = sum(comb(n_ij, 2) for n_ij in contingency_matrix.flatten())

    # Calculate the expected index (as if the agreement is purely random)
    expected_index = sum_comb_c * sum_comb_k / comb(contingency_matrix.sum(), 2)
    max_index = (sum_comb_c + sum_comb_k) / 2
    ari = (sum_comb - expected_index) / (max_index - expected_index)

    return ari

def jarvis_patrick2(data, smin, k):
    distances = compute_distances(data)
    n = len(data)
    clusters = np.zeros(n, dtype=int)
    
    for i in range(n):
        neighbors = np.argsort(distances[i])[:k]
        for j in neighbors:
            if j == i:
                continue
            shared_neighbors = len(np.intersect1d(np.where(distances[i] <= smin), np.where(distances[j] <= smin)))
            if shared_neighbors >= smin:
                if clusters[i] == 0 and clusters[j] == 0:
                    clusters[i] = clusters[j] = np.max(clusters) + 1
                elif clusters[i] == 0:
                    clusters[i] = clusters[j]
                elif clusters[j] == 0:
                    clusters[j] = clusters[i]
                else:
                    min_cluster = min(clusters[i], clusters[j])
                    clusters[clusters == clusters[i]] = min_cluster
                    clusters[clusters == clusters[j]] = min_cluster

    unique_clusters, counts = np.unique(clusters, return_counts=True)
    while len(unique_clusters) > 5:
        min_cluster = unique_clusters[np.argmin(counts)]
        clusters[clusters == min_cluster] = 0
        unique_clusters, counts = np.unique(clusters, return_counts=True)
    
    new_clusters = np.zeros(len(clusters), dtype=int)
    for i, cluster in enumerate(unique_clusters):
        new_clusters[clusters == cluster] = i + 1
        print(f"Cluster {i + 1}: {counts[i]} points")

    return new_clusters

def jarvis_patrick(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[NDArray[np.int32] | None, float | None, float | None]:
    """
    Implementation of the Jarvis-Patrick algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'k', 'smin', There could be others.
    - params_dict['k']: the number of nearest neighbors to consider. This determines the size of the neighborhood used to assess the similarity between datapoints. Choose values in the range 3 to 8
    - params_dict['smin']:  the minimum number of shared neighbors to consider two points in the same cluster.
       Choose values in the range 4 to 10.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index

    Notes:
    - the nearest neighbors can be bidirectional or unidirectional
    - Bidirectional: if point A is a nearest neighbor of point B, then point B is a nearest neighbor of point A).
    - Unidirectional: if point A is a nearest neighbor of point B, then point B is not necessarily a nearest neighbor of point A).
    - In this project, only consider unidirectional nearest neighboars for simplicity.
    - The metric  used to compute the the k-nearest neighberhood of all points is the Euclidean metric
    """
    smin= int(params_dict["smin"])
    k = int(params_dict["k"])
    distances = compute_distances(data)
    n = len(data)
    neighbors  = np.argsort(distances, axis=1)[:, 1:k+1]
    shared_neighbors = np.zeros_like(distances)
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            shared_count = np.intersect1d(neighbors[i], neighbors[j]).shape[0]
            if shared_count >= smin:
                shared_neighbors[i, j] = shared_neighbors[j, i] = 1
    unvisited = set(range(len(data)))
    clusters = []
    while unvisited:
        # Randomly pick an unvisited point
        point = unvisited.pop()
        cluster = [point]
        points_to_visit = set(np.where(shared_neighbors[point] == 1)[0])
        while points_to_visit:
            point = points_to_visit.pop()
            if point in unvisited:
                unvisited.remove(point)
                cluster.append(point)
                new_neighbors = set(np.where(shared_neighbors[point] == 1)[0])
                points_to_visit |= new_neighbors
        clusters.append(cluster)
    computed_labels = np.zeros(len(data), dtype=int)
    for idx, cluster in enumerate(clusters):
        computed_labels[cluster] = idx
    #clusters = np.zeros(n, dtype=int)
    #print("params",params_dict)
    #smin= int(params_dict["smin"])
    #k = int(params_dict["k"])
    SSE = 0
    for cluster in clusters:
        points = data[cluster]
        centroid = np.mean(points, axis=0)
        SSE += np.sum((points - centroid) ** 2)
    ARI = calculate_ari(labels, computed_labels)
    """
    for i in range(n):
        neighbors = np.argsort(distances[i])[:k]
        for j in neighbors:
            if j == i:
                continue
            shared_neighbors = len(np.intersect1d(np.where(distances[i] <= smin), np.where(distances[j] <= smin)))
            if shared_neighbors >= smin:
                if clusters[i] == 0 and clusters[j] == 0:
                    clusters[i] = clusters[j] = np.max(clusters) + 1
                elif clusters[i] == 0:
                    clusters[i] = clusters[j]
                elif clusters[j] == 0:
                    clusters[j] = clusters[i]
                else:
                    min_cluster = min(clusters[i], clusters[j])
                    clusters[clusters == clusters[i]] = min_cluster
                    clusters[clusters == clusters[j]] = min_cluster

    unique_clusters, counts = np.unique(clusters, return_counts=True)
    while len(unique_clusters) > 5:
        min_cluster = unique_clusters[np.argmin(counts)]
        clusters[clusters == min_cluster] = 0
        unique_clusters, counts = np.unique(clusters, return_counts=True)
    
    new_clusters = np.zeros(len(clusters), dtype=int)
    for i, cluster in enumerate(unique_clusters):
        new_clusters[clusters == cluster] = i + 1
        print(f"Cluster {i + 1}: {counts[i]} points")
    computed_labels=new_clusters
    SSE=compute_sse(data, clusters)
    ARI = calculate_ari(labels, clusters-1)
    
    computed_labels: NDArray[np.int32] | None = None
    SSE: float | None = None
    ARI: float | None = None
    """

    return computed_labels, SSE, ARI


def jarvis_patrick_clustering():
    """
    Performs Jarvis-Patrick clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}
    groups = []
    # Return your `jarvis_patrick` function
    answers["jarvis_patrick_function"] = jarvis_patrick

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Jarvis-Patrick.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    
    data=np.load('question1_cluster_data.npy')
    labels=np.load('question1_cluster_labels.npy')
    n_points = data.shape[0]
    indices = np.random.choice(n_points, size=5000, replace=False)
    selected_data = data[indices]
    selected_labels = labels[indices]
    print(selected_data.shape)
    print(selected_labels.shape)
    list_i=[0,1,2,3,4]
    slice={}
    slice_labels={}
    for i in list_i:
        slice[i]=selected_data[i*1000:(i+1)*1000]
        slice_labels[i]=selected_labels[i*1000:(i+1)*1000]
    print("slice0",slice[0].shape) 
    print("slice0labels",slice_labels[0].shape) 
    k_values = np.linspace(3, 8, 5)
    smin_values = np.linspace(4, 10, 5)
    #smin_values = [4,6,8,10]
    #k_values = [3,5,7,8]
    #smin_values = [0.15]
    #k_values = [10]
    groups=[]
    sse_list1=[]
    ari_list1=[]
    best_smin = None
    best_ari = -1
    best_k = None
    for smin in smin_values:
        for k in k_values:
            params_dict = {'k': k, 'smin': smin}
            computed_labels, SSE, ARI = jarvis_patrick(slice[0], slice_labels[0],params_dict)
            groups.append( {"smin": smin, "k":k,"ARI": ARI, "SSE": SSE})
    groups = {i: {'smin': group['smin'], 'k':group['k'],'ARI': group['ARI'], 'SSE': group['SSE']} for i, group in enumerate(groups)}
    """
    for smin in smin_values:
        for k in k_values:
            clusters = jarvis_patrick2(slice[0], smin, k)
            sse = compute_sse(slice[0], clusters)
            ari = calculate_ari(slice_labels[0], clusters-1)
            groups.append( {"smin": smin, "k":k,"ARI": ari, "SSE": sse})
            sse_list1.append(sse)
            ari_list1.append(ari)
            print(f"smin={smin}, k={k}, SSE={sse}, ARI={ari}")
            if ari > best_ari:
                best_ari = ari
                best_smin = smin
                best_k = k
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
            markers = ['o', 's', '^', 'D', 'x']
            n_clusters = len(np.unique(clusters))
            plt.figure(figsize=(8, 6))
            for i in range(1, n_clusters +1 ):
                cluster_points = slice[0][clusters == i]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i % len(colors)], marker=markers[i % len(markers)], label=f'Cluster {i}')

            plt.title(f'Cluster Visualization (smin={smin}, k={k})')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.show()
    print(f"Best ARI: {best_ari}, smin={best_smin}, k={best_k}")
""" 
    max_ari2 = -float('inf')
    target_sse = None
    for i in range(len(groups)):
            if groups[i]["ARI"] > max_ari2:
                max_ari2 = groups[i]["ARI"]
                target_sse = groups[i]["SSE"]
                best_smin=groups[i]["smin"]
                best_k=groups[i]["k"]
    average_sse = 0
    average_ari = 0
    sse_list2=[]
    ari_list2=[]
    for i in range(5):
        smin=best_smin
        k=best_k
        params_dict = {'k': k, 'smin': smin}
        computed_labels, SSE, ARI = jarvis_patrick(slice[i], slice_labels[i],params_dict)
        average_sse += SSE
        average_ari += ARI
        sse_list2.append(SSE)
        ari_list2.append(ARI)
        
        
    average_sse /= 5  # Divide by the number of slices
    average_ari /= 5  # Divide by the number of slices
    std_sse=np.std(sse_list2)
    std_ari=np.std(ari_list2)
    
    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups[0]['SSE']

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with # \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    # Plot is the return value of a call to plt.scatter()
    plot_ARI = plt.scatter([1,2,3], [4,5,6])
    plot_SSE = plt.scatter([1,2,3], [4,5,6])
    
    smin_values = [groups[i]['smin'] for i in groups]
    k_values = [groups[i]['k'] for i in groups]
    sse_values = [groups[i]['SSE'] for i in groups]
    ari_values= [groups[i]['ARI'] for i in groups]
    plt.figure(figsize=(10, 6))
    plot_SSE=plt.scatter(smin_values, k_values, c=sse_values, cmap='viridis')
    plt.colorbar(label='SSE')
    plt.xlabel('smin')
    plt.ylabel('k')
    plt.title('SSE for Different Parameter Values')
    plt.show()
    print("sseok")
    plt.figure(figsize=(10, 6))
    plot_ARI=plt.scatter(smin_values, k_values, c=ari_values, cmap='viridis')
    plt.colorbar(label='ARI')
    plt.xlabel('smin')
    plt.ylabel('k')
    plt.title('ARI for Different Parameter Values')
    plt.show()
    print("ariok")
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.

    # A single float
    answers["mean_ARIs"] = average_ari

    # A single float
    answers["std_ARIs"] = std_ari

    # A single float
    answers["mean_SSEs"] = average_sse

    # A single float
    answers["std_SSEs"] = std_sse
    print("answer ok")
    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = jarvis_patrick_clustering()
    with open("jarvis_patrick_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
