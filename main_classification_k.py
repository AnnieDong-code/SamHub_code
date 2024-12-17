import numpy as np
from skhubness.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import scipy
import math
import os
import h5py
from scipy.spatial import distance
import faiss
from scipy import stats
from skhubness import Hubness
from skhubness.reduction import LocalScaling
from ANN_Ninh import ANN_ours
from tslearn.neighbors import KNeighborsTimeSeries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import os
from tqdm import tqdm
from SamHub import Algorithm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from scipy.stats import norm
from hub_toolbox.hubness_analysis import SEC_DIST

'''
=========================================================
A. Compare with different hubness reduction methods:
(1) nicdm or local scaling (ls)
(2) dis_sim_local (dsl): AAAI-17;
(3) lcent: AAAI-15;

B. Verification on kNN with different methods:
(1)~(3) same as before
(4) weighted SamHub
'''

def compute_accuracy(exact_neighbors, approx_neighbors):
    total = exact_neighbors.shape[0]
    correct = 0
    for i in range(total):
        correct += len(np.intersect1d(exact_neighbors[i], approx_neighbors[i]))
    accuracy = correct / (total * exact_neighbors.shape[1])
    return accuracy

def computeKNN(q, x, k, metric):
    faiss.omp_set_num_threads(4)
    if metric == 'L1':
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='manhattan').fit(x)
        return nbrs.kneighbors(q)
    elif metric == 'L2' or 'euclidean':
        # faiss.omp_set_num_threads(-1)
        index = faiss.IndexFlatL2(x.shape[1])
        index.add(x)
        distances, indices = index.search(q, k)
        return distances, indices
    elif metric == 'dot':
        index = faiss.IndexFlatIP(x.shape[1])
        index.add(x)
        distances, indices = index.search(q, k)
        return distances, indices
    elif metric == 'cosine':
        knn = NearestNeighbors(n_neighbors=k, metric='cosine')
        knn.fit(x)
        return knn.kneighbors(q)
    elif metric == 'DTW':

        knn = KNeighborsTimeSeries(n_neighbors=k, metric="dtw", n_jobs=1)
        knn.fit(x)

        return knn.kneighbors(q)

    else:
        raise ValueError("Unsupported metric. Use 'L1', 'L2', 'dot', or 'cosine'.")


#---------Load data----------
current_dir = os.path.dirname(os.path.abspath(__file__))
data_name = 'ISOLET'  # Arrhythmia_452_279_13; Gisette_6000_5000_2; Optical Recognition of Handwritten Digits_5619_63_10;
file_name = 'ISOLET_7796_616_26'  # CNAE-9_1080_856_9  Mini-newsgroups_4_1353_27226_4
file_path = os.path.join(current_dir, 'Datasets', f'reallife datasets\\{file_name}.npz')
dataset = np.load(file_path)
X = dataset['X']
y = dataset['y']
n, d = X.shape
sf = 0.1

print(f'Load {file_name}: shape of X:{n}; num of X:{n};')

#---------Set para----------
metric = 'euclidean'  # 'cosine' None('euclidean')
n_jobs = -1
k_hubness = 50
k_classification = 10

print(f'Paras: dis={metric}, k_hubness={k_hubness}, k_classification={k_classification}')


# Main
print('----------------exact------------------------')
disMatrix, indMatrix = computeKNN(X, X, k_hubness, metric)
counter = np.zeros(n)
for topK in indMatrix:
    for x in topK:
        counter[x] = counter[x] + 1

""" Calculate exact skewness"""
exact_skewness = stats.skew(counter)
print("Skewness of exact kNN: ", exact_skewness)
# moment3 = scipy.stats.moment(counter, moment=3, axis=0, nan_policy='propagate')
# std3 = math.pow(np.std(counter), 3)
# print("Exact_Skewness_2: ", moment3 / std3)
#
# hub = Hubness(k=k, return_value='k_skewness', metric=metric)
# hub.fit(X)
# score = hub.score()
# print("Exact_Skewness_3: ", score)

""" Calculate accuracy on exact kNN"""
knn_exact = KNeighborsClassifier(n_neighbors=k_classification, algorithm='brute', n_jobs=None)
knn_exact.fit(X, y)
y_pred_exact = knn_exact.predict(X)
accuracy_exact = accuracy_score(y, y_pred_exact)
print("Accuracy of exact kNN: ", accuracy_exact)


# print('----------------ls/nicdm (packsge)------------------------')
# """ Calculate nn using hubness reduction"""
# nn_nicdm = NearestNeighbors(n_neighbors=k_hubness,
#                             metric=metric,
#                             algorithm='brute',
#                             algorithm_params={'n_candidates': k_hubness,
#                                               'n_jobs': -1},
#                             hubness='nicdm',
#                             verbose=0,
#                             )
# nn_nicdm.fit(X)
# pred_nn_nicdm = nn_nicdm.kneighbors(X, n_neighbors=k_classification, return_distance=False)
#
# """ Calculate skewness after hubness reduction"""
# counter_nicdm = np.zeros(n)
# for topK in pred_nn_nicdm:
#     for x in topK:
#         counter_nicdm[x] = counter_nicdm[x] + 1
# nicdm_skewness = stats.skew(counter_nicdm)
# print(f"Skewness after 'nicdm': ", nicdm_skewness)
#
# """ Verification on kNN after hubness reduction"""
# y_pred_nicdm = np.zeros(pred_nn_nicdm.shape[0], dtype=int)
# for i, neighbors in enumerate(pred_nn_nicdm):
#     neighbor_labels = y[neighbors]
#     y_pred_nicdm[i] = stats.mode(neighbor_labels)[0][0]
# accuracy_nicdm = accuracy_score(y, y_pred_nicdm)
# print("Accuracy after 'nicdm': ", accuracy_nicdm)


print(f'----------------ls/nicdm (hand)------------------------')
disMatrix, indMatrix = computeKNN(X, X, k_hubness, metric)
""" Step 1: calculate mu_x and mu_y """
r_train = disMatrix.mean(axis=1)
r_test = disMatrix.mean(axis=1)

""" Step 2: recalculate dist by nicdm """
hub_reduced_dist = np.empty_like(disMatrix)
for i in range(n):
    hub_reduced_dist[i, :] = disMatrix[i] / np.sqrt(r_test[i] * r_train[indMatrix[i]])

""" Step 3: update kNN """
dehub_neigh_ind = np.argsort(hub_reduced_dist, axis=1)[:, :k_classification]
dehub_neigh_ind = np.take_along_axis(indMatrix, dehub_neigh_ind, axis=1)

""" Calculate skewness after hubness reduction"""
counter_nicdm = np.zeros(n)
for topK in dehub_neigh_ind:
    for x in topK:
        counter_nicdm[x] = counter_nicdm[x] + 1
nicdm_skewness = stats.skew(counter_nicdm)
print(f"Skewness after 'nicdm': ", nicdm_skewness)

""" Verification on kNN after hubness reduction"""
y_pred_nicdm = np.zeros(dehub_neigh_ind.shape[0], dtype=int)
for i, neighbors in enumerate(dehub_neigh_ind):
    neighbor_labels = y[neighbors]
    y_pred_nicdm[i] = stats.mode(neighbor_labels)[0][0]
accuracy_nicdm = accuracy_score(y, y_pred_nicdm)
print("Accuracy after 'nicdm': ", accuracy_nicdm)


# print('----------------dsl (package)------------------------')
# """ Calculate nn using hubness reduction"""
# nn_dsl = NearestNeighbors(n_neighbors=k_hubness,
#                             metric=metric,
#                             algorithm='brute',
#                             algorithm_params={'n_candidates': k_hubness,
#                                               'n_jobs': -1},
#                             hubness='dsl',
#                             verbose=0,
#                             )
# nn_dsl.fit(X)
# pred_nn_dsl = nn_dsl.kneighbors(X, n_neighbors=k_classification, return_distance=False)
#
# """ Calculate skewness after hubness reduction"""
# counter_dsl = np.zeros(n)
# for topK in pred_nn_dsl:
#     for x in topK:
#         counter_dsl[x] = counter_dsl[x] + 1
# dsl_skewness = stats.skew(counter_dsl)
# print(f"Skewness after 'dsl': ", dsl_skewness)
#
# """ Verification on kNN after hubness reduction"""
# y_pred_dsl = np.zeros(pred_nn_dsl.shape[0], dtype=int)
# for i, neighbors in enumerate(pred_nn_dsl):
#     neighbor_labels = y[neighbors]
#     y_pred_dsl[i] = stats.mode(neighbor_labels)[0][0]
# accuracy_dsl = accuracy_score(y, y_pred_dsl)
# print("Accuracy after 'dsl': ", accuracy_dsl)


print('----------------dsl (hand)------------------------')
# Step 1: Compute the k nearest neighbors (KNN) using brute force or a provided method
disMatrix, indMatrix = computeKNN(X, X, k_hubness, metric)

# Step 2: Compute the local centroids for each sample based on the k-nearest neighbors
centroids = np.zeros((X.shape[0], X.shape[1]))

for i in range(X.shape[0]):
    neighbors = indMatrix[i, :k_hubness]  # Get the k-nearest neighbors indices
    centroids[i] = X[neighbors].mean(axis=0)  # Calculate centroid as the mean of neighbors

# Step 3: Compute the distance of each point to its local centroid
dist_to_centroids = np.linalg.norm(X - centroids, axis=1) ** 2

# Step 4: Adjust the distances using DisSimLocal
hub_reduced_dist = np.empty_like(disMatrix)

for i in range(X.shape[0]):
    for j in range(disMatrix.shape[1]):
        # Calculate the adjusted distance using the formula in DisSimLocal
        hub_reduced_dist[i, j] = disMatrix[i, j] - dist_to_centroids[i] - dist_to_centroids[indMatrix[i, j]]

# Step 5: Ensure non-negative distances by shifting negative values up
min_dist = hub_reduced_dist.min()
if min_dist < 0:
    hub_reduced_dist -= min_dist

# Step 6: Sort neighbors based on the reduced distances and get the updated indices
dehub_neigh_ind = np.argsort(hub_reduced_dist, axis=1)[:, :k_classification]
dehub_neigh_dist = np.take_along_axis(hub_reduced_dist, dehub_neigh_ind, axis=1)
dehub_neigh_ind = np.take_along_axis(indMatrix, dehub_neigh_ind, axis=1)

# Step 7: Calculate skewness after hubness reduction
counter_disSimLocal = np.zeros(X.shape[0])
for topK in dehub_neigh_ind:
    for x in topK:
        counter_disSimLocal[x] += 1
disSimLocal_skewness = stats.skew(counter_disSimLocal)
print(f"Skewness after 'dsl': ", disSimLocal_skewness)

# Step 8: Verification of kNN classification after hubness reduction
y_pred_disSimLocal = np.zeros(dehub_neigh_ind.shape[0], dtype=int)
for i, neighbors in enumerate(dehub_neigh_ind):
    neighbor_labels = y[neighbors]
    y_pred_disSimLocal[i] = stats.mode(neighbor_labels)[0][0]

accuracy_disSimLocal = accuracy_score(y, y_pred_disSimLocal)
print("Accuracy after 'dsl': ", accuracy_disSimLocal)


print('----------------lcent (package)------------------------')
""" Calculate nn using hubness reduction (Localized Centering)"""
# Call lcent using the SEC_DIST dictionary
# lcent works with vectors (X), so we pass X instead of the distance matrix
D_lcent = 1.0 - SEC_DIST['lcent'](X)  # 1.0 - to transform similarity to distance

# Apply KNN on the lcent distance matrix
nn_lcent = NearestNeighbors(n_neighbors=k_hubness, metric=metric, n_jobs=n_jobs)
nn_lcent.fit(D_lcent)
pred_nn_lcent = nn_lcent.kneighbors(D_lcent, n_neighbors=k_classification, return_distance=False)

""" Calculate skewness after hubness reduction (Localized Centering)"""
counter_lcent = np.zeros(n)
for topK in pred_nn_lcent:
    for x in topK:
        counter_lcent[x] = counter_lcent[x] + 1
lcent_skewness = stats.skew(counter_lcent)
print(f"Skewness after 'lcent': ", lcent_skewness)

""" Verification on kNN after hubness reduction (Localized Centering)"""
y_pred_lcent = np.zeros(pred_nn_lcent.shape[0], dtype=int)
for i, neighbors in enumerate(pred_nn_lcent):
    neighbor_labels = y[neighbors]
    y_pred_lcent[i] = stats.mode(neighbor_labels)[0][0]
accuracy_lcent = accuracy_score(y, y_pred_lcent)
print("Accuracy after 'lcent': ", accuracy_lcent)


# print('----------------lcent (hand)------------------------')
# # Step 1: Normalize the vectors to unit length
# norms = np.linalg.norm(X, axis=1, keepdims=True)
# norms[norms == 0] = 1e-7  # Avoid division by zero
# X_normalized = X / norms
#
# # Step 2: Compute the similarity matrix using dot products
# similarity_matrix = np.dot(X_normalized, X_normalized.T)
#
# # Step 3: Calculate local centroids and affinities
# local_affinity = np.zeros(X.shape[0])
#
# for i in range(X.shape[0]):
#     # Find the kappa-nearest neighbors by similarity
#     neighbors = np.argpartition(similarity_matrix[i, :], -k_hubness)[-k_hubness:]
#     # Calculate local centroid as the mean of kappa nearest neighbors
#     local_centroid = X_normalized[neighbors].mean(axis=0)
#     # Compute local affinity as the dot product of sample and local centroid
#     local_affinity[i] = np.dot(X_normalized[i], local_centroid)
#
# # Step 4: Adjust the similarity matrix by subtracting local affinities
# adjusted_similarity = similarity_matrix - local_affinity[:, np.newaxis]
#
# # Step 5: Calculate skewness of hubness reduction
# counter_lcent = np.zeros(X.shape[0])
# for i in range(X.shape[0]):
#     # Get the kappa-nearest neighbors based on the adjusted similarity
#     topK = np.argpartition(adjusted_similarity[i, :], -k_hubness)[-k_hubness:]
#     for x in topK:
#         counter_lcent[x] += 1
#
# # Skewness after localized centering
# lcent_skewness = stats.skew(counter_lcent)
# print(f"Skewness after 'lcent': ", lcent_skewness)
#
# # Step 6: Verification of kNN classification after localized centering
# y_pred_lcent = np.zeros(X.shape[0], dtype=int)
# for i in range(X.shape[0]):
#     # Find kappa-nearest neighbors
#     neighbors = np.argpartition(adjusted_similarity[i, :], -k_classification)[-k_classification:]
#     neighbor_labels = y[neighbors]
#     # Majority voting
#     y_pred_lcent[i] = stats.mode(neighbor_labels)[0][0]
#
# accuracy_lcent = accuracy_score(y, y_pred_lcent)
# print("Accuracy after 'lcent': ", accuracy_lcent)


print('----------------Removed hubs------------------------')
seeds = np.random.randint(0, 1000, size=5)
print(f'sf={sf}; random seed: {seeds}')
accuracy_nohubs_list = []
for seed in seeds:
    _, _, _, approx_hubs_t2, _ = Algorithm(X, X, s=int(sf * X.shape[0]), k=k_hubness, h=10, metric=metric, sd=456)
    X_nohubs = np.delete(X, approx_hubs_t2, axis=0)
    y_nohubs = np.delete(y, approx_hubs_t2)
    knn_nohubs = KNeighborsClassifier(n_neighbors=k_classification, algorithm='brute', metric=metric, n_jobs=None)
    knn_nohubs.fit(X_nohubs, y_nohubs)
    y_pred_nohubs = knn_nohubs.predict(X)
    accuracy_nohubs = accuracy_score(y, y_pred_nohubs)
    accuracy_nohubs_list.append(accuracy_nohubs)

Samhub_Removed_accuracy = np.mean(accuracy_nohubs_list)
print("Average accuracy of removed kNN: ", Samhub_Removed_accuracy)


print('----------------weighted kNN------------------------')
seeds = np.random.randint(0, 1000, size=5)
print(f'sf={sf}; random seed: {seeds}')
accuracy_weighted_list = []
for seed in seeds:
    _, _, _, _, counter = Algorithm(X, X, s=int(sf * X.shape[0]), k=k_hubness, h=10, metric=metric, sd=456)

    hubs_BNk = counter / int(sf * 100)
    mu_hubs_BNk = np.mean(hubs_BNk)
    sigma_hubs_BNk = np.std(hubs_BNk)
    hubs_h_B = (hubs_BNk - mu_hubs_BNk) / sigma_hubs_BNk
    hubs_weights_train = np.exp(-hubs_h_B)

    # Step 3: 使用加权投票进行分类预测
    knn = KNeighborsClassifier(n_neighbors=k_classification, algorithm='brute', n_jobs=None)
    knn.fit(X, y)

    neighbors = knn.kneighbors(X, return_distance=False)
    y_pred_weighted = []

    for i, test_neighbors in enumerate(neighbors):
        class_votes = {}
        for idx in test_neighbors:
            label = y[idx]
            weight = hubs_weights_train[idx]
            if label in class_votes:
                class_votes[label] += weight
            else:
                class_votes[label] = weight
        y_pred_weighted.append(max(class_votes, key=class_votes.get))

    y_pred_weighted = np.array(y_pred_weighted)
    accuracy_weighted = accuracy_score(y, y_pred_weighted)
    accuracy_weighted_list.append(accuracy_weighted)

Samhub_Weighted_accuracy = np.mean(accuracy_weighted_list)
print("Average accuracy of weighted kNN: ", Samhub_Weighted_accuracy)


save_dir = r'D:\Dong_Files\PhD_project\Python\Hubness\Fast_Hubs\res\main_classification_k'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

filename = f'results_{data_name}_{k_hubness}_{k_classification}_sf_{sf}.txt'
file_path = os.path.join(save_dir, filename)

results = []

results.append(f"Skewness of exact: {exact_skewness}")
results.append(f"Accuracy of exact: {accuracy_exact}")

results.append(f"Skewness of nicdm: {nicdm_skewness}")
results.append(f"Accuracy of nicdm: {accuracy_nicdm}")

results.append(f"Skewness of dsl: {disSimLocal_skewness}")
results.append(f"Accuracy of dsl: {accuracy_disSimLocal}")

results.append(f"Skewness of lcent: {lcent_skewness}")
results.append(f"Accuracy of lcent: {accuracy_lcent}")


results.append(f"Accuracy of noHubs: {accuracy_nohubs}")
results.append(f"Average of SamHub: {Samhub_Weighted_accuracy}")

with open(file_path, 'w') as f:
    f.write(f"sf={sf}, k_hubness={k_hubness}, k_classification={k_classification}\n")
    f.write("Seeds: " + ", ".join(map(str, seeds)) + "\n\n")
    for result in results:
        f.write(result + "\n")

print(f"Results saved to {file_path}")
