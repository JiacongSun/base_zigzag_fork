import logging
import pickle
import numpy as np
from api import density_extraction_across_channel_dim, derive_model_layer_count
import time
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_sparsity_matrix(tensor):
    """
    Calculate sparsity for each spatial position across channels
    Input tensor shape: 1 x 256 x 14 x 14 (for example)
    Returns: 14 x 14 matrix of sparsity values
    """
    # Remove batch dimension and transpose to get (14, 14, 256)
    activations = tensor[0].transpose(1, 2, 0)

    (x_size, y_size, c_size) = activations.shape

    # Calculate sparsity (fraction of zero values) for each spatial position
    sparsity_matrix = np.zeros((x_size, y_size))
    for i in range(x_size):
        for j in range(y_size):
            channel_values = activations[i, j, :]
            channel_values_binary = []
            for value in channel_values:
                channel_values_binary.append(1 if value > 0 else 0)
            channel_values_binary = np.array(channel_values_binary)
            sparsity = np.sum(channel_values_binary == 0) / c_size
            sparsity_matrix[i, j] = sparsity

    return sparsity_matrix


# Assuming your four tensors are named tensor1, tensor2, tensor3, tensor4
def plot_sparsity_patterns(tensors):
    """
    Create four subplots showing sparsity patterns for each tensor
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    # fig.suptitle('Sparsity Patterns Across Spatial Positions\n(Fraction of zero values across 256 channels)',
    #              fontsize=16)

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    # Calculate global min and max for consistent colorbar scale
    all_sparsity_values = []
    for tensor in tensors:
        sparsity_matrix = calculate_sparsity_matrix(tensor)
        all_sparsity_values.extend(sparsity_matrix.flatten())
    vmin, vmax = min(all_sparsity_values), max(all_sparsity_values)

    for idx, (ax, tensor) in enumerate(zip(axes_flat, tensors)):
        sparsity_matrix = calculate_sparsity_matrix(tensor)

        # Calculate average sparsity and std for this tensor
        avg_sparsity = np.mean(sparsity_matrix)
        std_sparsity = np.std(sparsity_matrix)

        # Create heatmap
        sns.heatmap(sparsity_matrix,
                    ax=ax,
                    cmap='viridis',  # Yellow-Orange-Red colormap
                    vmin=vmin,
                    vmax=vmax,
                    cbar_kws={'label': 'Sparsity'})
        # # use matplotlib.colorbar.Colorbar object
        # cbar = ax.collections[0].colorbar
        # # here set the labelsize
        # cbar.ax.tick_params(labelsize=30)
        ax.figure.axes[-1].yaxis.label.set_size(12)

        ax.set_title(
            f'Case {idx + 1} ' + '(Sparsity: $\mu$:' + f'{avg_sparsity:.3f}, ' + '$\sigma$:' + f'{std_sparsity:.3f})',
        loc='left',
        pad=10, weight='bold', fontsize=14)
        ax.set_xlabel('OX', fontsize=12, weight='bold')
        ax.set_ylabel('OY', fontsize=12, weight='bold')

        # Add box around the subplot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)  # Adjust the line width of the box
            spine.set_color('black')  # Set the color of the box

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)
    plt.show()


if __name__ == "__main__":
    """
    demonstrating activation density in pictures
    """
    logging_level = logging.INFO  # logging level
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)
    ##############################
    ## parameters
    model_name = "resnet18"
    dataset_name = "imagenet"
    layer_idx = 2
    channel_size = 256  # not useful in the plotting in this script
    ##############################
    # generate image indices
    # if dataset_name == "cifar10":
    #     img_indices = np.random.randint(0, 10000, size=5)
    # else:  # imagenet
    #     img_indices = np.random.randint(1, 40000, size=5)
    img_indices = [27860, 23407, 8978, 27411]  # fix the indices
    layer_count = derive_model_layer_count(model_name)
    assert layer_idx < layer_count
    START_TIME = time.time()
    # extract density information
    tensor_collect: list
    density_mean_collect: list
    density_std_collect: list
    density_covariance_matrix: np.ndarray
    tensor_collect = density_extraction_across_channel_dim(
        tile_size=channel_size,
        layer_idx=layer_idx,
        img_indices=img_indices,
        model_name=model_name,
        dataset_name=dataset_name)
    plot_sparsity_patterns(tensor_collect)
    # timing report
    END_TIME = time.time()
    time_in_second = round(END_TIME - START_TIME, 2)
    logging.debug(f"Total extraction time (seconds): {time_in_second}")
