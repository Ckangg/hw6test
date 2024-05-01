"""
Work with Spectral clustering.
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

def  proximity_measure(distance,sigma):
    W = np.exp(-distance ** 2 / (2 * sigma ** 2))
    return W
def construct_similarity_matrix(data, sigma):
    n_samples=data.shape[0]
    similarity_matrix= np.zeros((n_samples,n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            distance=np.linalg.norm(data[i]-data[j])
            similarity_matrix[i,j]=proximity_measure(distance,sigma)
    return similarity_matrix
def compute_l(W):
    D=np.diag(np.sum(W,axis=1))
    L=D-W
    return L 
number_clusters=5
def calculate_eigens(L_matrix,number_clusters):
    eigenvalues,eigenvectors=np.linalg.eigh(L_matrix)
    return eigenvectors[:, :number_clusters]
iterations = 100
def k_means(data,number_clusters,iterations):
    indices= np.random.choice(data.shape[0],number_clusters,replace=False)
    centroids=data[indices]
    for _ in range(iterations):
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        closest_centroid = np.argmin(distances, axis=0)
        for i in range(number_clusters):
            centroids[i] = data[closest_centroid == i].mean(axis=0)
    return closest_centroid,centroids
def compute_sse(data, cluster_labels, centroids):
    sse = 0.0
    for i, centroid in enumerate(centroids):
        cluster_data = data[cluster_labels == i]
        if cluster_data.size > 0:
            sse += np.sum((cluster_data - centroid) ** 2)
        else:
            print(f"Warning: Cluster {i} has no points assigned.")
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

def spectral(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[
    NDArray[np.int32] | None, float | None, float | None, NDArray[np.floating] | None
]:
    """
    Implementation of the Spectral clustering  algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'sigma', and 'k'. There could be others.
       params_dict['sigma']:  in the range [.1, 10]
       params_dict['k']: the number of clusters, set to five.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index
    - eigenvalues: eigenvalues of the Laplacian matrix
    """
    sigma = params_dict['sigma']
    k = params_dict['k']
    k =5
    number_clusters=k
    smatrix=construct_similarity_matrix(data,sigma)
    L_matrix=compute_l(smatrix)
    eigenvalues,eigenvectors_L=np.linalg.eigh(L_matrix)
    normalized_eigenvectors = eigenvectors_L / np.linalg.norm(eigenvectors_L, axis=1, keepdims=True)
    computed_labels,centroids = k_means(normalized_eigenvectors, number_clusters, iterations)
    SSE = compute_sse(normalized_eigenvectors, labels, centroids)
    #print(f"sigma:{sigma_list[i]}SSE {sse}:")
    ARI =calculate_ari(labels,computed_labels)
    #print(f"sigma:{sigma_list[i]}ARI {ari}:")
    """computed_labels: NDArray[np.int32] | None = None
    SSE: float | None = None
    ARI: float | None = None
    eigenvalues: NDArray[np.floating] | None = None"""

    return computed_labels, SSE, ARI, eigenvalues


def spectral_clustering():
    """
    Performs DENCLUE clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}
    groups = {}

    # Return your `spectral` function
    
    answers["spectral_function"] = spectral
    data = np.load("C:\\Users\\11203\\pyStudy\\TranFraud\\pgm\\CAP-5771-s24-hw6-main\\CAP-5771-s24-hw6-main\\question1_cluster_data.npy")
    labels = np.load("C:\\Users\\11203\\pyStudy\\TranFraud\\pgm\\CAP-5771-s24-hw6-main\\CAP-5771-s24-hw6-main\\question1_cluster_labels.npy")
    n_points = data.shape[0]
    indices = np.random.choice(n_points, size=5000, replace=False)

    selected_data = data[indices]
    selected_labels = labels[indices]
    #print(selected_data.shape)
    #print(selected_labels.shape)
    list_i=[0,1,2,3,4]
    slice={}
    slice_labels={}
    for i in list_i:
        slice[i]=selected_data[i*1000:(i+1)*1000]
        slice_labels[i]=selected_labels[i*1000:(i+1)*1000]
    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Spectral clustering.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').
    sigma_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    sse_list1=[]
    ari_list1=[]
    max_ari = float('-inf')
    best_sigma = None
    for i in range(10):
        sigma=sigma_list[i]
        smatrix=construct_similarity_matrix(slice[0],sigma)
        L_matrix=compute_l(smatrix)
        eigenvectors_L=calculate_eigens(L_matrix,number_clusters)
        normalized_eigenvectors = eigenvectors_L / np.linalg.norm(eigenvectors_L, axis=1, keepdims=True)
        clusters,centroids = k_means(normalized_eigenvectors, number_clusters, iterations)
        sse = compute_sse(normalized_eigenvectors, slice_labels[0], centroids)
        #print(f"sigma:{sigma_list[i]}SSE {sse}:")
        ari=calculate_ari(slice_labels[0],clusters)
        sse_list1.append(sse)
        ari_list1.append(ari)
        #print(f"sigma:{sigma_list[i]}ARI {ari}:")
        groups[i]= {"sigma": sigma, "ARI": ari, "SSE": sse}
        if ari > max_ari:
            max_ari = ari
            best_sigma = sigma
    print(f"The sigma value with the highest ARI is: {best_sigma} (ARI: {max_ari})")
    params_dict = {
    'sigma':0.2,  # Creates 100 values from 0.1 to 10 evenly spaced
    'k': 5  # Number of clusters
}
    max_ari2 = -float('inf')
    target_sse = None
    for i in range(len(groups)):
        if groups[i]["ARI"] > max_ari2:
            max_ari2 = groups[i]["ARI"]
            target_sse = groups[i]["SSE"]
    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    average_sse = 0
    average_ari = 0
    sse_list2=[]
    ari_list2=[]
    for i in range(5):
        sigma=params_dict['sigma']
        smatrix=construct_similarity_matrix(slice[i],sigma)
        L_matrix=compute_l(smatrix)
        eigenvectors_L=calculate_eigens(L_matrix,number_clusters)
        normalized_eigenvectors = eigenvectors_L / np.linalg.norm(eigenvectors_L, axis=1, keepdims=True)
        clusters,centroids = k_means(normalized_eigenvectors, number_clusters, iterations)
        sse = compute_sse(normalized_eigenvectors, slice_labels[i], centroids)
        print(f"slice:{i} SSE {sse}:")
        ari=calculate_ari(slice_labels[i],clusters)
        print(f"slice:{i} ARI {ari}:")
        sse_list2.append(sse)
        ari_list2.append(ari)
        average_sse += sse
        average_ari += ari
        unique_clusters, counts = np.unique(clusters, return_counts=True)
        print("Cluster distribution:")
        for cluster, count in zip(unique_clusters, counts):
            print(f"Cluster {cluster}: {count} points")

        plt.figure(figsize=(10, 6))
        for cluster in unique_clusters:
            # Extract points belonging to the current cluster
            cluster_points = slice[i][clusters == cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')

        plt.title(f'{i}Cluster Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        plt.show()
    average_sse /= 5  # Divide by the number of slices
    average_ari /= 5  # Divide by the number of slices

    print(f"Average SSE: {average_sse}")
    print(f"Average ARI: {average_ari}")
    std_sse=np.std(sse_list2)
    std_ari=np.std(ari_list2)
    smatrix=construct_similarity_matrix(slice[0],sigma)
    L_matrix=compute_l(smatrix)
    teigenvalues,_=np.linalg.eigh(L_matrix)
    #groups = {}

    # For the spectral method, perform your calculations with 5 clusters.
    # In this cas,e there is only a single parameter, σ.

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = {target_sse}

    # Identify the cluster with the lowest value of ARI. This implies
    # that you set the cluster number to 5 when applying the spectral
    # algorithm.

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    # Plot is the return value of a call to plt.scatter()
    #plot_ARI = plt.scatter([1,2,3], [4,5,6])
    #plot_SSE = plt.scatter([1,2,3], [4,5,6])
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_ARI = plt.scatter(sigma_list, ari_list1, c='blue', label='ARI Values')
    plt.xlabel('Sigma')
    plt.ylabel('ARI')
    plt.title('ARI Values vs. Sigma')
    plt.legend()
    plt.grid(True)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_SSE = plt.scatter(sigma_list, sse_list1, c='blue', label='SSE Values')
    plt.xlabel('Sigma')
    plt.ylabel('SSE')
    plt.title('SSE Values vs. Sigma')
    plt.legend()
    plt.grid(True)
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # Plot of the eigenvalues (smallest to largest) as a line plot.
    # Use the plt.plot() function. Make sure to include a title, axis labels, and a grid.
    #plot_eig = plt.plot([1,2,3], [4,5,6])
    plt.figure()
    plot_eig = plt.plot(range(1, len(teigenvalues) + 1), sorted(teigenvalues))
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalues (smallest to largest)')
    plt.grid(True)
    answers["eigenvalue plot"] = plot_eig

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

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = spectral_clustering()
    with open("spectral_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)