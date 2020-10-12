import numpy as np



MAX_ITERATIONS = 10
n_clusters = 2
X = np.random.uniform(1, 0, size=(3,3))


def k_means(X, n_clusters):
    centroids = np.eye(n_clusters, X.shape[1])
    # centroids = np.array([[0, 2, 0], [1, 1, 1]])
    print(centroids)
    for i in range(MAX_ITERATIONS):
        print("Iteration # {}".format(i))
        centroids, cluster_ids = k_means_loop(X, centroids)
        print(centroids)
    return centroids, cluster_ids


def k_means_loop(X, centroids):
    # find labels for rows in X based in centroids values
    expanded_centroids = centroids[:, None]
    distances = np.sqrt(np.sum((expanded_centroids - X) ** 2, axis=2))
    arg_min = np.argmin(distances, axis=0)
    # recompute centroids
    for i in range(centroids.shape[0]):
        centroids[i] = np.mean(X[arg_min == i, :], axis=0)
    return centroids, arg_min


centroid, cluster_id = k_means(X, n_clusters)
