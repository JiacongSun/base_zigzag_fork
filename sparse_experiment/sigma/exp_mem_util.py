import math
import numpy as np
import matplotlib.pyplot as plt
import scipy


def derive_idx_precision(encoding, tile_size, dense_element_counts, average_density):
    # calc idx precision
    if encoding == "coo":
        idx_precision_within_tile = math.log2(tile_size)
        idx_precision_across_tile = math.log2(dense_element_counts / tile_size)
        idx_precision = idx_precision_within_tile + idx_precision_across_tile
    elif encoding == "csr" or encoding == "csc":
        sparse_element_counts = dense_element_counts * average_density
        idx_precision_within_tile = math.log2(tile_size)
        idx_precision_across_tile = math.log2(sparse_element_counts) * (dense_element_counts / tile_size + 1) / sparse_element_counts
        idx_precision = idx_precision_within_tile + idx_precision_across_tile
    elif encoding == "bm":
        idx_precision = dense_element_counts / (dense_element_counts * average_density)
    else:
        idx_precision = 0
    return idx_precision


def plot_distribution_vs_val(val: list, mean: list, std: list):
    assert len(val) == len(mean) == len(std)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(figsize=(12, 8))

    for idx in range(len(val)):
        x = val[idx]
        mu = mean[idx]
        sigma = std[idx]
        x_range = np.linspace(0, 1, 200)  # range for distribution values
        # Plot distributions for each x value
        distribution = scipy.stats.norm.pdf(x_range, mu, sigma)
        # Scale the distribution for visibility and shift it based on x
        scaled_dist = distribution * 0.1  # Scale factor for visibility
        # Plot filled distribution
        ax.fill_between(x_range, x, x + scaled_dist, alpha=0.3, color=colors[int(math.log2(x))])

        # Plot distribution outline
        ax.plot(x_range, x + scaled_dist, 'k-', linewidth=0.5)
    # Customize plot
    ax.set_xlabel('Memory utilization', fontsize=12, weight='bold')
    ax.set_ylabel('Sz_tile', fontsize=12, weight='bold')

    # Add colorbar to show density
    # norm = plt.Normalize(x_range.min(), x_range.max())
    # sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    # sm.set_array([])
    # plt.colorbar(sm, label='$Sz_{tile}$')

    sz_level = list(set(val))
    plt.yticks(sz_level, [int(2**x) for x in sz_level])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """
    Exp: mem utilization (512 KB) @ L2, ResNet18
    """
    mem_size_512kb = 512 * 1024 * 8
    ox = 56
    oy = 56
    c = 64
    op_pres = 8
    encoding = "csr"
    tile_size = 8
    average_density = 0.6
    density_std = 0.09

    encoding_pool = ["csr", "coo", "bm"]
    tile_size_pool = [4, 8, 16, 32, 64, 128]
    # for mem utilization
    util_mean_pool = []
    util_std_pool = []
    # for required memory footprint (byte)
    mf_mean_pool = []
    mf_std_pool = []

    for encoding in encoding_pool:
        for tile_size in tile_size_pool:
            dense_element_counts = ox * oy * c
            idx_precision = derive_idx_precision(encoding, tile_size, dense_element_counts, average_density)
            size_occupied_bit = dense_element_counts * average_density * (op_pres + idx_precision)
            curr_util: dict = {
                "mean": size_occupied_bit / mem_size_512kb,
                "std": size_occupied_bit / average_density * density_std / mem_size_512kb,
                }
            memory_footprint: dict = {
                "mean": size_occupied_bit / 8,
                "std": dense_element_counts * density_std * (op_pres + idx_precision) / 8,
            }

            print(f"encoding: {encoding}, Sz_tile: {tile_size}, idx: {idx_precision}, required_mem_byte: {size_occupied_bit/8}, util_mean: {curr_util['mean']}, util_std: {curr_util['std']}, 6std: {6 * curr_util['std']}")
            util_mean_pool.append(curr_util["mean"])
            util_std_pool.append(curr_util["std"])
            mf_mean_pool.append(memory_footprint["mean"])
            mf_std_pool.append(memory_footprint["std"])

    tile_size_pool = [math.log2(x) for x in tile_size_pool]
    plot_distribution_vs_val(val=tile_size_pool*len(encoding_pool), mean=util_mean_pool, std=util_std_pool)
    # plot_distribution_vs_val(val=tile_size_pool, mean=mf_mean_pool, std=mf_std_pool)  # not attractive
