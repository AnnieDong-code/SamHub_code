import numpy as np
import os
from SamHub import Algorithm, exactKNN
import matplotlib.pyplot as plt


# Load and prepare data
current_dir = os.path.dirname(os.path.abspath(__file__))
file_name = 'Splice_1000_61_2'
data_name = 'Splice'
file_path = os.path.join(current_dir, f'Datasets\\{file_name}.npz')
dataset = np.load(file_path, allow_pickle=True)
X = dataset['X']
X /= np.linalg.norm(X, axis=1).reshape(-1, 1)

# Set parameters
metric = 'euclidean'
sf = 0.1
s = int(sf * X.shape[0])
h = 10  # Get top-h hubs
k_values = [5, 10, 15, 20, 25]

# Store results
skewness_results = {}
hub_overlap_results = {}
candi_size_results = {}

random_seeds = np.random.randint(0, 10000, size=3)

for k in k_values:
    print(k)
    _, exact_skew, exact_hubs = exactKNN(X, k, h, metric)

    approx_skew_t2_list = []
    hub_overlap_t2_list = []

    for seed in random_seeds:
        _, _, approx_skew_t2, approx_hubs_t2, _, _ = Algorithm(X, X, s=s, k=k, h=h, metric=metric, sd=seed)

        approx_skew_t2_list.append(approx_skew_t2)
        # candi_size_list.append(candi_size)

        hub_overlap_t2 = len(np.intersect1d(exact_hubs, approx_hubs_t2)) / h

        hub_overlap_t2_list.append(hub_overlap_t2)

    avg_approx_skew_t2 = np.mean(approx_skew_t2_list)
    avg_hub_overlap_t2 = np.mean(hub_overlap_t2_list)

    skewness_results[k] = [exact_skew, avg_approx_skew_t2]
    hub_overlap_results[k] = [avg_hub_overlap_t2]


#==============plot
fig, ax1 = plt.subplots(figsize=(5, 4))

bar_width = 0.3
index = np.arange(len(k_values))

# Define colors
exact_color = '#6F6F6F'
approx_color = '#547BB4'
accuracy_color = 'red'

# Plot skewness results
rects1 = ax1.bar(index - bar_width/2, [skewness_results[k][0] for k in k_values], bar_width,
                 label='Exact', color=exact_color, alpha=0.7, hatch='//')
rects2 = ax1.bar(index + bar_width/2, [skewness_results[k][1] for k in k_values], bar_width,
                 label='SamHub', color=approx_color, alpha=0.7)

# Annotate bars with the values from the right y-axis
for i, (rect1, rect2, value) in enumerate(zip(rects1, rects2, [hub_overlap_results[k][0] for k in k_values])):
    higher_bar_height = max(rect1.get_height(), rect2.get_height())
    ax1.text(index[i], higher_bar_height + 0.05, f'Acc:\n{100*value:.2f}%', ha='center', va='bottom',
             color=accuracy_color, fontweight=500)

# Set labels and title
ax1.set_xlabel('k')
ax1.set_ylabel('Skewness')
# plt.title(f'Skewness and Hub Detection Accuracy in {data_name}')

# Set x-axis ticks
ax1.set_xticks(index)
ax1.set_xticklabels([str(k) for k in k_values])

# Set the y-axis limits to give space for annotations
old_ylim1 = ax1.get_ylim()
ax1.set_ylim(old_ylim1[0], old_ylim1[1] + 0.1 * (old_ylim1[1] - old_ylim1[0]))

ax1.set_xticklabels([str(k) for k in k_values])

# Add legends inside the plot
ax1.legend(loc='best')

# Adjust layout and save
plt.tight_layout()
plt.savefig(f'Utility_k_{data_name}_s_{sf}_h_{h}.eps', format='eps', bbox_inches='tight')
plt.savefig(f'Utility_k_{data_name}_s_{sf}_h_{h}.png', format='png', bbox_inches='tight')

plt.show()