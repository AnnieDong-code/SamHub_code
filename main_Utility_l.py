import numpy as np
import os
from SamHub import Algorithm, exactKNN
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde


# Load and prepare data
current_dir = os.path.dirname(os.path.abspath(__file__))
file_names = [
    'Splice_1000_61_2',
]
data_names = [
    'Splice',
]

# Placeholder for skewness results
skewness_results = []

for file_name, data_name in zip(file_names, data_names):
    print(file_name)
    file_path = os.path.join(current_dir, f'Datasets\\{file_name}.npz')
    dataset = np.load(file_path)
    X = dataset['X']

    # Normalize
    n, d = X.shape

    # Set parameters
    metric = 'euclidean'
    s = int(0.1 * n)
    k = 10  # kNN
    h = 20  # Get top-h hubs
    seed = 36

    # Calculate freq
    exact_all_freq, _, exact_hubs = exactKNN(X, k, h, metric)
    _, _, _, approx_hubs_t2, _, approx_all_freq = Algorithm(X, X, s=s, k=k, h=h, metric=metric, sd=seed)

    # Find points with zero frequency in SamHub
    zero_freq_indices = np.where(approx_all_freq == 0)[0]

    # Create a mapping from original indices to new indices after deletion
    index_mapping = np.delete(np.arange(len(approx_all_freq)), zero_freq_indices)

    # Map original hub indices to their new positions after removing zero frequencies
    exact_hubs_mapped = np.array(
        [np.where(index_mapping == hub)[0][0] for hub in exact_hubs if hub not in zero_freq_indices])
    approx_hubs_t2_mapped = np.array(
        [np.where(index_mapping == hub)[0][0] for hub in approx_hubs_t2 if hub not in zero_freq_indices])

    # Delete zero frequency points in exact_all_freq and approx_all_freq
    exact_all_freq = np.delete(exact_all_freq, zero_freq_indices, axis=0)
    approx_all_freq = np.delete(approx_all_freq, zero_freq_indices, axis=0)

    # Divide SamHub frequency by 10
    approx_all_freq /= 10

#---------------------------
    # Accuracy calculation
    accurate_hubs = np.intersect1d(exact_hubs_mapped, approx_hubs_t2_mapped)
    accuracy = 100 * len(accurate_hubs) / h
    print(f"Accuracy of hubs: {accuracy:.2f}%")

    # Spearman correlation calculation
    correlation, _ = stats.spearmanr(exact_all_freq, approx_all_freq)
    print(f"Spearman correlation of counts: {correlation:.4f}")

#--------- Plotting PDF
    # exact_all_freq_norm = exact_all_freq - np.mean(exact_all_freq)
    # approx_all_freq_norm = approx_all_freq - np.mean(approx_all_freq)

    exact_all_freq_norm = (exact_all_freq - np.mean(exact_all_freq)) / np.std(exact_all_freq)
    approx_all_freq_norm = (approx_all_freq - np.mean(approx_all_freq)) / np.std(approx_all_freq)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    exact_all_freq_norm = scaler.fit_transform(exact_all_freq.reshape(-1, 1)).flatten()
    approx_all_freq_norm = scaler.fit_transform(approx_all_freq.reshape(-1, 1)).flatten()

    kde_exact = gaussian_kde(exact_all_freq_norm)  # bw_method=0.5
    x_exact = np.linspace(min(exact_all_freq_norm), max(exact_all_freq_norm), 1000)
    pdf_exact = kde_exact(x_exact)

    kde_approx = gaussian_kde(approx_all_freq_norm)
    x_approx = np.linspace(min(approx_all_freq_norm), max(approx_all_freq_norm), 1000)
    pdf_approx = kde_approx(x_approx)

    plt.figure(figsize=(5, 4))
    plt.plot(x_exact, pdf_exact, label='Exact kNN', color='red')
    plt.plot(x_approx, pdf_approx, label='SamHub', color='SeaGreen', linestyle='--')
    plt.xlabel('Frequency')
    plt.ylabel('Probability Density')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'Utility_pdf_{data_name}.eps', format='eps', bbox_inches='tight')
    plt.savefig(f'Utility_pdf_{data_name}.png', format='png', bbox_inches='tight')

    plt.show()

#--------- Plotting PDF
    exact_all_freq_norm = (exact_all_freq - np.mean(exact_all_freq))
    approx_all_freq_norm = (approx_all_freq - np.mean(approx_all_freq))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4), sharex=False)
    fig.subplots_adjust(hspace=0.0)  # Remove space between subplots

    # Calculate KDE for both datasets
    kde_exact = gaussian_kde(exact_all_freq_norm)
    kde_approx = gaussian_kde(approx_all_freq_norm)

    # Generate points for plotting
    x_exact = np.linspace(min(exact_all_freq_norm), max(exact_all_freq_norm), 1000)
    x_approx = np.linspace(min(approx_all_freq_norm), max(approx_all_freq_norm), 1000)

    # Calculate densities
    density_exact = kde_exact(x_exact)
    density_approx = kde_approx(x_approx)

    # Plot for Exact kNN (top)
    ax1.plot(x_exact, density_exact, color='red')
    ax1.fill_between(x_exact, density_exact, 0, alpha=0.5, color='red')
    ax1.set_ylabel('Density (Exact kNN)', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.set_xlabel('Frequency (Exact kNN)', color='red')
    ax1.xaxis.set_label_position('top')
    ax1.xaxis.tick_top()

    # Plot for SamHub (bottom)
    ax2.plot(x_approx, density_approx, color='SeaGreen')
    ax2.fill_between(x_approx, density_approx, 0, alpha=0.5, color='SeaGreen')
    ax2.set_ylabel('Density (SamHub)', color='SeaGreen')
    ax2.tick_params(axis='y', labelcolor='SeaGreen')
    ax2.set_xlabel('Frequency (SamHub)', color='SeaGreen')

    # Set y-axis ticks and limits
    y_max_exact = np.max(density_exact)
    y_max_approx = np.max(density_approx)

    ax1.set_ylim(0, y_max_exact)
    ax2.set_ylim(y_max_approx, 0)  # Inverted for bottom plot

    # Set 5 ticks for each y-axis, including 0
    ax1_ticks = np.linspace(0, y_max_exact, 5)
    ax2_ticks = np.linspace(0, y_max_approx, 5)

    ax1.set_yticks(ax1_ticks)
    ax2.set_yticks(ax2_ticks)

    # Format tick labels to avoid scientific notation
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

    # Highlight the 0 tick in black
    ax1.get_yticklabels()[0].set_color('black')
    # ax2.get_yticklabels()[-1].set_color('black')

    # Remove the middle spines
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(bottom=False)
    ax2.tick_params(top=False)

    # Add a horizontal line at y=0 for both plots to create a connected look
    ax1.axhline(y=0, color='black', linestyle='-', lw=2)
    ax2.axhline(y=0, color='black', linestyle='-', lw=2)
    plt.savefig(f'Utility_pdf_{data_name}.eps', format='eps', bbox_inches='tight')
    plt.savefig(f'Utility_pdf_{data_name}.png', format='png', bbox_inches='tight')

    plt.show()


# --------- Plotting Freq
    plt.figure(figsize=(5, 4))
    plt.plot(np.arange(len(exact_all_freq)), exact_all_freq, label='Exact', linestyle='-',
             color='red', alpha=1)
    plt.plot(np.arange(len(approx_all_freq)), approx_all_freq, label='SamHub', linestyle='-',
             color='SeaGreen', alpha=1)

    # Marking approximate hubs
    for hub in approx_hubs_t2_mapped:
        plt.plot(hub, approx_all_freq[hub], 'o', markerfacecolor='SeaGreen', markeredgewidth=2,
                 markeredgecolor='SeaGreen', markersize=6)

    # Marking exact hubs
    for hub in exact_hubs_mapped:
        plt.plot(hub, exact_all_freq[hub], 's', markerfacecolor='red', markeredgewidth=2, markeredgecolor='red',
                 markersize=6)

    # Highlight accurate predictions with black border
    for hub in accurate_hubs:
        plt.plot(hub, approx_all_freq[hub], 'o', color='SeaGreen', markeredgecolor='black', markeredgewidth=2,
                 markersize=6)

    # Mark exact hubs not included in approx hubs
    inaccurate_hubs = np.setdiff1d(exact_hubs_mapped, approx_hubs_t2_mapped)
    for hub in inaccurate_hubs:
        plt.plot(hub, exact_all_freq[hub], 's', markerfacecolor='red', markeredgewidth=2, markeredgecolor='red',
                 markersize=6)
        plt.plot(hub, exact_all_freq[hub], 'x', color='black', markeredgewidth=2, markersize=8)

    # Set y-axis range
    max_value = max(max(exact_all_freq), max(approx_all_freq)) + 5
    plt.ylim(0, max_value)
    yticks = np.linspace(0, max_value, 6)
    plt.yticks(yticks)

    # Custom legend for each method
    custom_legend = [
        plt.Line2D([0], [0], color='red', marker='s', linestyle='-', markersize=6),
        plt.Line2D([0], [0], color='SeaGreen', marker='o', linestyle='-', markersize=6),
    ]
    # custom_legend = [
    #     plt.Line2D([0], [0], color='red', linestyle='-', markersize=6),
    #     plt.Line2D([0], [0], color='SeaGreen', linestyle='-', markersize=6),
    # ]

    plt.legend(custom_legend, ['Exact', 'SamHub'])

    plt.xlabel('Index')
    plt.ylabel('Frequency')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'res\\main_Utility_l\\Utility_freq_h{h}_{data_name}.eps', format='eps', bbox_inches='tight')
    plt.savefig(f'res\\main_Utility_l\\Utility_freq_h{h}_{data_name}.png', format='png', bbox_inches='tight')

    plt.show()