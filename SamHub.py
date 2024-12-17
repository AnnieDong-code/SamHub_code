import numpy as np
from scipy import stats
import faiss
from sklearn.neighbors import NearestNeighbors
from tslearn.neighbors import KNeighborsTimeSeries


def exactKNN(X, k, h, metric):
    # ----------------------------------------------------------
    """ Find exact kNN """
    # First compute X * X^T, then sort
    disMatrix, indMatrix = computeKNN(X, X, k, metric)

    """ Find exact hubs """
    # First init counter, then counting
    counter = np.zeros(X.shape[0])
    for topK in indMatrix:
        for x in topK:
            counter[x] = counter[x] + 1

    # Now get hub by sorting and return max
    exact_hub_idx = np.argsort(-counter)[:h]

    exact_skewness = stats.skew(counter)
    return counter, exact_skewness, exact_hub_idx


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


def Algorithm(XI, XQ, s, k, h, metric, sd=None):
    """ Step 1 """
    # First sampling s element
    if sd != None:
        np.random.seed(sd)
    sampled_idx = np.random.choice(XI.shape[0], s)

    # Compute kNN between sample point
    S = XI[sampled_idx]

    dist, index = computeKNN(S, XI, k, metric)

    # Now find the approximate hubs by counting
    counter = np.zeros(XI.shape[0])
    candHub = set()
    for topK in index:
        for x in topK:
            counter[x] = counter[x] + 1
            candHub.add(x)  # add into candidate hub

    # Convert candHub to a list and sort by counter values
    candHub = list(candHub)
    candHub.sort(key=lambda x: counter[x], reverse=True)
    candHub = np.array(candHub)

    # Select top c of candHub
    top_c_candHub = candHub[:s]

    # Top-h hubs
    hub_idx_t1 = np.argsort(-counter)[:h]
    # Skewness on counter
    skewness_t1 = stats.skew(counter)

    """ Step 2 """
    C = XI[top_c_candHub]

    dist, index = computeKNN(XQ, C, k, metric)
    matTopK = candHub[index]

    # Now find the approximate hubs by counting
    counter = np.zeros(XI.shape[0])
    RkNN_C = {i: [] for i in range(XI.shape[0])}  # Initialize RkNN_C for all points
    for i, topK in enumerate(matTopK):
        for x in topK:
            counter[x] += 1
            RkNN_C[i].append(x)

    # Top-h hubs
    hub_idx_rank = np.argsort(counter)
    hub_idx_t2 = hub_idx_rank[-3*h:]
    # Skewness on counter
    skewness_t2 = stats.skew(counter)


    """ Step 3: Calculate overlap rate for hubs """
    overlap_rates = []

    for hub in hub_idx_t2:
        rknn = set(RkNN_C[hub])  # Reverse k-NN of the hub

        # Get the reverse k-NN of each point in rknn
        rknn_of_rknn = [set(RkNN_C[point]) for point in rknn]

        unique_elements = set().union(*rknn_of_rknn)
        unique_count = len(unique_elements)

        overlap_rates.append(unique_count)

    # Calculate standardized "bad" hubness score
    overlap_rates = -np.array(overlap_rates)  # higher unique_count = lower overlap_rates
    BN_k_mean = np.mean(overlap_rates)
    BN_k_std = np.std(overlap_rates)

    if BN_k_std == 0:
        h_B = np.zeros_like(overlap_rates)
    else:
        h_B = (overlap_rates - BN_k_mean) / BN_k_std

    # Compute weights
    weights = np.exp(h_B)  # higher weights = higher overlap_rates = higher pro hubs being center

    """ Step 4: Calculate weighted scores for hub_idx_t2 """
    weighted_scores = counter[hub_idx_t2] * weights
    sorted_weighted_scores = np.argsort(weighted_scores)
    sorted_weighted_scores = sorted_weighted_scores[-h:]
    hub_idx_t2_cluster = hub_idx_t2[sorted_weighted_scores]

    # print(f'candi_size: {s}')

    return skewness_t1, hub_idx_t1, skewness_t2, hub_idx_t2[-h:], hub_idx_t2_cluster, counter



# # Load data
# X = np.loadtxt('dexter_X.txt')
#
# # Normalize data
# X /= np.linalg.norm(X, axis=1).reshape(-1, 1)
#
# n, d = X.shape
#
# k = 10  # kNN
# h = 10  # Get top-h hubs
#
# #----------------------------------------------------------
# """ Find exact kNN """
# # First compute X * X^T, then sort
# dotMatrix = np.matmul(X, X.transpose())
# matTopK = np.zeros((n, k), dtype=int)
# for i in range(n):
#     # First sorting, then get index of max
#     matTopK[i] = np.argsort(-dotMatrix[i])[:k]  # topK MIPS indexes for each x
#
# """ Find exact hubs """
# # First init counter, then counting
# counter = np.zeros(n)
# for topK in matTopK:
#     for x in topK:
#         counter[x] = counter[x] + 1
#
# # Now get hub by sorting and return max
# exact_hub_idx = np.argsort(-counter)[:h]
#
# # Skewness on counter
# moment3 = scipy.stats.moment(counter, moment=3, axis=0, nan_policy='propagate')
# std3 = math.pow(np.std(counter), 3)
# print("Exact Skewness: ", moment3 / std3)
#
# #----------------------------------------------------------
#
#
# """ Test 1: Fast identifying hubs
#
# Fact:
# - We compute hubs by executing n kNN queries where each query is a point x in X
# - Complexity: O(dn^2)
#
# Hypothesis:
# - Can we compute hubs by executing s << n kNN queries ?
# - We sample s points from X and use them as s queries to identify hubs
# - Complexity O(dsn)
# - Strength: This method is generic, working with any distance measures (cosine, L1, L2, ...)
# """
#
# # First sampling s element
# s = 5  # np.ceil(math.sqrt(n)).astype('int')
# sampled_idx = np.random.choice(n, s)
#
# # Compute kNN between sample point
# S = X[sampled_idx]
# dotMatrix = np.matmul(S, X.transpose())
#
# # This is kNN of s samples
# matTopK = np.zeros((s, k), dtype=int)
# for i in range(s):
#     # First sorting, then get index of max
#     matTopK[i] = np.argsort(-dotMatrix[i])[:k]  # topK MIPS indexes for each sample s
#
# # Now find the approximate hubs by counting
# counter = np.zeros(n)
# candHub = set()
# for topK in matTopK:
#     for x in topK:
#         counter[x] = counter[x] + 1
#         candHub.add(x)  # add into candidate hub
#
# candHub = np.array(list(candHub))  # convert to numpy array
#
# # Skewness
# moment3 = scipy.stats.moment(counter, moment=3, axis=0, nan_policy='propagate')
# std3 = math.pow(np.std(counter), 3)
# print("Approx skewness of Test 1: ", moment3 / std3)
#
# # Now get hub by sorting and return max
# hub_idx_t1 = np.argsort(-counter)[:h]
# print("Accuracy of Test1: ", len(np.intersect1d(exact_hub_idx, hub_idx_t1)) / h)
# print("Intersection size between hub and candHub: ", len(np.intersect1d(exact_hub_idx, candHub)) / h)
#
#
# #----------------------------------------------------------
#
#
# """ Test 2: Fast identifying hubs
#
# Fact:
# - Given s samples, we form the candidate hub of size at most sk
# - These points tend to contain the exact hubs, though its frequency is not dominating the other points in candHub.
# - If we compute kNN(x, candHub), the exact hubs in candHub tend to appear on kNN list
#
# Hypothesis:
# - Can we extract hubs from the candidate hub?
# - We compute kNN(x, candHub) for all point x in X. The exact hub will show up with high probability.
# - Complexity O(dsk) + O(dn sk)
# - Strength: This method is generic, working with any distance measures (cosine, L1, L2, ...)
# """
#
# # Get the point from the candidate hub
# # candHub = np.sort(candHub)
# S = X[candHub]
#
# # Compute dot product between X and candidate hub
# dotMatrix = np.matmul(X, S.transpose())
#
# # This is kNN of n points, each kNN points is from the candHub
# matTopK = np.zeros((n, k), dtype=int)
# for i in range(n):
#     # First sorting, then get index of max
#     matTopK[i] = candHub[
#         np.argsort(-dotMatrix[i])[:k]]  # Note that, np.argsort(-dotMatrix[i])[:1] return index on S, not on X
#
# # Now find the approximate hubs by counting
# counter = np.zeros(n)
# for topK in matTopK:
#     for x in topK:
#         counter[x] = counter[x] + 1
#
# # Now get hub by sorting and return max
# hub_idx_t2 = np.argsort(-counter)[:h]
#
# # Skewness on counter
# moment3 = scipy.stats.moment(counter, moment=3, axis=0, nan_policy='propagate')
# std3 = math.pow(np.std(counter), 3)
# print("Approx skewness of Test2: ", moment3 / std3)
#
# # This accuracy should nearly be the same as intersection size between hub and candHub
# print("Accuracy of Test2: ", len(np.intersect1d(exact_hub_idx, hub_idx_t2)) / h)
