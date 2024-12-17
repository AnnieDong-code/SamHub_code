import numpy as np
import matplotlib.pyplot as plt
import os
from SamHub import Algorithm
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

def exponential_smoothing(BNk, alpha=0.3):
    smoothed_BNk = np.zeros_like(BNk)
    smoothed_BNk[0] = BNk[0]
    for t in range(1, len(BNk)):
        smoothed_BNk[t] = alpha * BNk[t] + (1 - alpha) * smoothed_BNk[t-1]
    return smoothed_BNk

current_dir = os.path.dirname(os.path.abspath(__file__))
file_names = [
    'Dexter_300_20000_2.npz',
]
data_names = [
    'Dexter',
]

# Set parameters
method = 'KNN'
metric = 'euclidean'
sf = 0.1
h = 10
k_values = [5, 10, 15, 20, 25]

random_seeds = np.random.randint(0, 1000, size=5)
print(random_seeds)


for dataset, data_name in zip(file_names, data_names):
    print(data_name)
    # Init
    classification_accuracies = []
    execution_times = []

    # Load data
    file_path = os.path.join(current_dir, f'Datasets\\{dataset}')
    data = np.load(file_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    X /= np.linalg.norm(X, axis=1).reshape(-1, 1)
    s = int(sf * X.shape[0])

    accuracy_exact_all = []
    accuracy_weighted_all = []
    accuracy_hubs_removed_all = []
    accuracy_hubs_weighted_all = []

    time_exact_all = []
    time_weighted_all = []
    time_hubs_removed_all = []
    time_hubs_weighted_all = []

    for k in k_values:
        print(k)

        for seed in random_seeds:
            # Split the data into training and testing sets
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
            # # No Split
            X_train, X_test = X, X
            y_train, y_test = y, y

            # ==========Exact KNN==================
            start_time = time.time()
            knn_exact = KNeighborsClassifier(n_neighbors=k, algorithm='brute', n_jobs=None)
            knn_exact.fit(X_train, y_train)
            y_pred_exact = knn_exact.predict(X_test)
            exact_time = time.time() - start_time
            accuracy_exact = accuracy_score(y_test, y_pred_exact)
            accuracy_exact_all.append(accuracy_exact)
            time_exact_all.append(exact_time)


            # ==========noLabel Weighted KNN==================
            # Calculate hubness scores and weights for the training data
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute', n_jobs=None).fit(X_train)
            distances, indices = nbrs.kneighbors(X_train)
            BNk = np.zeros(X_train.shape[0])

            for i in range(X_train.shape[0]):
                for j in indices[i, 1:]:  # Exclude itself
                    BNk[j] += 1

            mu_BNk = np.mean(BNk)
            sigma_BNk = np.std(BNk)
            h_B = (BNk - mu_BNk) / sigma_BNk
            weights_train = np.exp(-h_B)

            # Train and evaluate a KNN classifier using weights
            start_time = time.time()
            knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute', n_jobs=None)
            knn.fit(X_train, y_train)

            # Custom prediction with weights
            neighbors = knn.kneighbors(X_test, return_distance=False)
            y_pred = []
            for i, test_neighbors in enumerate(neighbors):
                class_votes = {}
                for idx in test_neighbors:
                    label = y_train[idx]
                    weight = weights_train[idx]
                    if label in class_votes:
                        class_votes[label] += weight
                    else:
                        class_votes[label] = weight
                y_pred.append(max(class_votes, key=class_votes.get))
            y_pred = np.array(y_pred)
            weighted_time = time.time() - start_time

            accuracy_weighted = accuracy_score(y_test, y_pred)
            accuracy_weighted_all.append(accuracy_weighted)
            time_weighted_all.append(weighted_time)


            #=============Hubs removed kNN==============
            # reset h
            # h = int(X_train.shape[0] / 30)
            start_time = time.time()
            _, _, _, approx_hubs_t2, approx_hubs_weighted, counter = (
                Algorithm(X_train, X_test, s=s,
                          k=k, h=h, metric=metric, sd=seed))

            # Remove the hubs from the training data
            hubs_idx = approx_hubs_t2  # approx_hubs_weighted
            X_train_remaining = np.delete(X_train, hubs_idx, axis=0)
            y_train_remaining = np.delete(y_train, hubs_idx, axis=0)

            # Train and evaluate a KNN classifier on the remaining training data
            start_time = time.time()
            knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute', n_jobs=None)
            knn.fit(X_train_remaining, y_train_remaining)
            y_pred = knn.predict(X_test)
            hubs_time = time.time() - start_time
            accuracy_hubs = accuracy_score(y_test, y_pred)
            accuracy_hubs_removed_all.append(accuracy_hubs)
            time_hubs_removed_all.append(hubs_time)


            # =============noLabel Hubs-based weighted kNN==============
            # Calculate weights based on frequency counter
            # hubs_BNk = exponential_smoothing(counter, alpha=0.3)
            hubs_BNk = counter/10
            mu_hubs_BNk = np.mean(hubs_BNk)
            sigma_hubs_BNk = np.std(hubs_BNk)
            hubs_h_B = (BNk - mu_hubs_BNk) / sigma_hubs_BNk
            hubs_weights_train = np.exp(-hubs_h_B)

            # Train and evaluate a KNN classifier using frequency-based weights
            start_time = time.time()
            knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute', n_jobs=None)
            knn.fit(X_train, y_train)

            # Custom prediction with frequency-based weights
            neighbors = knn.kneighbors(X_test, return_distance=False)
            y_pred_frequency = []
            for i, test_neighbors in enumerate(neighbors):
                class_votes = {}
                for idx in test_neighbors:
                    label = y_train[idx]
                    weight = hubs_weights_train[idx]
                    if label in class_votes:
                        class_votes[label] += weight
                    else:
                        class_votes[label] = weight
                y_pred_frequency.append(max(class_votes, key=class_votes.get))
            y_pred_frequency = np.array(y_pred_frequency)
            frequency_time = time.time() - start_time

            accuracy_hubs_weighted = accuracy_score(y_test, y_pred_frequency)
            accuracy_hubs_weighted_all.append(accuracy_hubs_weighted)
            time_hubs_weighted_all.append(frequency_time)


        # Calculate average accuracy and execution time
        classification_accuracies.append((k, np.mean(accuracy_exact_all), np.mean(accuracy_hubs_removed_all), np.mean(accuracy_hubs_weighted_all)))
        execution_times.append((k, np.mean(time_exact_all), np.mean(time_weighted_all), np.mean(time_hubs_weighted_all)))

    # ----------------Plot------------------------
    exact_color = 'black'
    hubs_removed_color = 'cyan'
    hubs_weighted_color = 'magenta'

    k_values = [acc[0] for acc in classification_accuracies]
    accuracies_exact = [acc[1] for acc in classification_accuracies]
    accuracies_hubs_removed = [acc[2] for acc in classification_accuracies]
    accuracies_hubs_weighted = [acc[3] for acc in classification_accuracies]

    plt.figure(figsize=(4, 3))
    plt.plot(k_values, accuracies_exact, marker='o', markerfacecolor='none', linestyle='-', color=exact_color,
             linewidth=2, label='Exact kNN')
    plt.plot(k_values, accuracies_hubs_removed, marker='+', linestyle='-', color=hubs_removed_color, linewidth=2,
             label='Removing hubs')
    plt.plot(k_values, accuracies_hubs_weighted, marker='x', linestyle='-', color=hubs_weighted_color, linewidth=2,
             label='weighted kNN')

    max_value = min(max(max(accuracies_exact), max(accuracies_hubs_removed), max(accuracies_hubs_weighted)) +0.01, 1)
    min_value = max(min(min(accuracies_exact), min(accuracies_hubs_removed), min(accuracies_hubs_weighted)) -0.01, 0)
    plt.ylim(min_value, max_value)

    yticks = np.linspace(min_value, max_value, 5)
    plt.yticks(yticks)

    plt.gca().set_yticklabels([f'{y:.3f}' for y in yticks])

    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.legend(fontsize=10, loc=(0.37, 0.47))   # loc=(0.5, 0.5)
    plt.grid(False)
    plt.tight_layout()

    # Save
    plt.savefig(f'Classification_k_h{h}_{data_name}.png', format='png')
    plt.savefig(f'Classification_k_h{h}_{data_name}.eps', format='eps')

    plt.show()
    # plt.close('all')


