import numpy as np

# n_samples = 10
# # definir una matriz con centroides
# centroids = np.array([[1, 0, 0, 0],[0, 1, 0, 0]])
# # Alejar centroides entre si con una constante
# centroids_far = centroids*2
# # crear n/2 muestras de cada centroide
# repeted_array=np.repeat(centroids_far,n_samples/2,axis=0)
# # sumar ruido iid
# vector_iid = np.random.normal(loc=0,scale=1,size=(n_samples,4))
# data=repeted_array+vector_iid
# # armar un arreglo con n enteros indicando si la muestra pertenece a A o B
# cluster_ids = np.array([
# [0],
# [1],
# ])
# cluster_ids = np.repeat(cluster_ids, n_samples / 2, axis=0)
# print(cluster_ids)
# print(data)


def build_cluster(n_samples,inv_overlap):
    centroids = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)
    centroids = centroids * inv_overlap
    data = np.repeat(centroids, n_samples / 2, axis=0)
    normal_noise = np.random.normal(loc=0, scale=1, size=(n_samples, 4))
    data = data + normal_noise
    cluster_ids = np.array([
        [0],
        [1],
    ])
    cluster_ids = np.repeat(cluster_ids, n_samples / 2, axis=0)
    return data, cluster_ids


data, cluster_id = build_cluster(10,3)
print(data)
print(cluster_id)
