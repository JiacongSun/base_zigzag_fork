import matplotlib.pyplot as plt
from inference_hooker import NetworkInference
import numpy as np
from scipy.stats import norm
from torchvision.io import read_image
import logging
import pickle
import torch as torch
from api import read_pickle, save_to_pickle
import random


def density_covariance_matrix_parser(
        density_list_collect: list,
        density_occurrence_collect: list,
):
    """
    calculate tile-level density covariance matrix across samples, for each layer
    :param density_list_collect: list[ndarray]: [[ll_sp0, ll_sp1, ..., ll_spn], [...], ...samples...]
    :param density_occurrence_collect: list[ndarray]: [[P_sp0, P_sp1, ..., P_spn], [...], ...samples...]
    :return density_covariance_matrix: ndarray
    """
    # extract density level (ll_density) count
    level_count = len(density_list_collect[0])
    # create array for calculating the mean of P_density product
    prob_product_mean_array: np.ndarray
    prob_product_mean_array = np.zeros((level_count, level_count))
    # create vector for calculating the mean of each P_density
    prob_mean_vector: np.ndarray
    prob_mean_vector = np.zeros(level_count)

    # covariance calculation
    sample_count = len(density_list_collect)
    for img_index in range(sample_count):
        for first_prob_index in range(level_count):
            p1_sample = density_occurrence_collect[img_index][first_prob_index]
            prob_mean_vector[first_prob_index] += p1_sample / sample_count
            for second_prob_index in range(first_prob_index, level_count):
                p2_sample = density_occurrence_collect[img_index][second_prob_index]
                p1p2_sample = p1_sample * p2_sample
                prob_product_mean_array[first_prob_index][second_prob_index] += p1p2_sample / sample_count
                if first_prob_index != second_prob_index:
                    prob_product_mean_array[second_prob_index][first_prob_index] += p1p2_sample / sample_count
    # calculate density covariance matrix
    density_covariance_matrix: np.ndarray
    density_covariance_matrix = prob_product_mean_array - np.outer(prob_mean_vector, prob_mean_vector)
    # density_covariance_matrix = np.round(density_covariance_matrix, 3)  # keep 3 decimal places
    return density_covariance_matrix


def extract_kernel_of_sparse_nn(layer_idx: int = 2,
                                model_name: str = "resnet18_sparse", ):
    """
    extract kernel (weights) of sparse networks
    :param layer_idx: targeted layer index to observe
    :param model_name: targeted inference model name, options: [resnet18_sparse, mobilenetv2_sparse]
    :return: kernels: tensor
    """
    assert model_name in ["resnet18_sparse", "mobilenetv2_sparse", "efficientnetb0_sparse"]

    folder_path = "./sparse_nn"
    if model_name == "resnet18_sparse":
        weights = torch.load(f"{folder_path}/best_rs18_per-tensor__distill__amanda_69_356.pth")
    elif model_name == "mobilenetv2_sparse":
        weights = torch.load(f"{folder_path}/mobilenetv2_quant_torchvision.pth")
    else:  # "efficientnet-b0", Year: 2020. Model is from the 1st one at: https://github.com/lukemelas/EfficientNet-PyTorch/releases
        weights = torch.load(f"{folder_path}/adv-efficientnet-b0-b64d5a18.pth")

    weight_dict = {}
    for k, v in weights.items():
        if ("conv" in k and "weight" in k):
            weight_dict[k] = {
                "layer_name": k,
                "float_value": torch.dequantize(v),
                # "int_value": v.int_repr(),  # efficientnet does not have this
                # "scale": v.q_scale(),  # efficientnet does not have this
                # "zero_point": v.q_zero_point()  # efficientnet does not have this
            }

    # layerwise weight fetching
    layer_count = len(weight_dict.keys())
    assert 0 <= layer_idx < layer_count
    layer_name = list(weight_dict.keys())[layer_idx]
    ans = weight_dict[layer_name]
    kernels = ans["float_value"]
    return kernels


def density_extraction_weight(tile_size: int = 8,
                              layer_idx: int = 2,
                              model_name: str = "resnet18", ):
    """
    Statically extract tile-level weight density distribution
    :param tile_size: targeted tile size
    :param layer_idx: targeted layer index to observe
    :param model_name: targeted inference model name, options: [resnet18, resnet50, vgg19, mobilenetv2, mobilenetv3, quant_mobilenetv2]
    :return: density_list: list of density levels (ll_den)
             density_occurrence: list of density occurrence
             density_mean: average tile-level density
             density_std: list of standard variance of the tile-level density per sample
    """
    density_list: np.ndarry
    density_occurrence: np.ndarry
    density_mean: float
    density_std: float
    if model_name in ["resnet18_sparse", "mobilenetv2_sparse", "efficientnetb0_sparse"]:
        # use sparse model from amanda
        kernels = extract_kernel_of_sparse_nn(
            layer_idx=layer_idx,
            model_name=model_name
        )
        if isinstance(kernels, torch.Tensor):
            kernels: np.ndarry = kernels.numpy()
        density_list, density_occurrence, density_mean, density_std = NetworkInference.calc_density_distribution(
            op_array=kernels,
            tile_size=tile_size,
            enable_relu=False)
    else:
        # use models inherited from pytorch
        # initialize the model
        nn = NetworkInference(model_name=model_name)
        # sparsity parsing
        intermediate_weight = nn.extract_weight_of_an_intermediate_layer(layer_idx=layer_idx)
        if isinstance(intermediate_weight, torch.Tensor):
            kernels: np.ndarry = intermediate_weight.numpy()
        # calculate tile-level density information
        density_list, density_occurrence, density_mean, density_std = NetworkInference.calc_density_distribution(
            op_array=intermediate_weight,
            tile_size=tile_size,
            enable_relu=False)
    return density_list, density_occurrence, density_mean, density_std


def density_extraction_with_fixed_img_indices(tile_size: int = 8,
                                              layer_idx: int = 2,
                                              img_indices: np.ndarray = np.random.randint(1, 10000, size=1),
                                              model_name: str = "resnet18",
                                              dataset_name: str = "imagenet",
                                              add_background: bool = True):
    """
    statically extract tile-level activation density distribution and density mean, with fixed img indices
    :param tile_size: targeted tile size
    :param layer_idx: targeted layer index to observe
    :param img_indices: image samples to observe
    :param model_name: targeted inference model name, options: [resnet18, resnet50, vgg19, mobilenetv2, mobilenetv3, quant_mobilenetv2]
    :param dataset_name: targeted dataset name, options: [cifar10, imagenet]
    :param add_background: consider 50% background images or not
    :return: density_list_collect: list of samples, each element containing density index of current sample
             density_occurrence_collect: list of samples, each element containing density occurrence of current sample
             aver_density_dist: dict, contains tile-level density information of the average distribution
             density_mean_collect: list of average tile-level density per sample
             density_std_collect: list of standard variance of the tile-level density per sample
    """
    # count image numbers
    img_numbers = len(img_indices)
    # initialize the model
    nn = NetworkInference(model_name=model_name, dataset_name=dataset_name)
    # record of density mean and std
    density_mean_collect = []
    density_std_collect = []  # not used
    # initialize average distribution: dict
    aver_density_dist = {i / tile_size: 0 for i in range(0, tile_size + 1)}
    # iterate across image indices
    illegal_count = 0
    # create pools for density information per sample
    density_list_collect = []  # density list per image sample
    density_occurrence_collect = []  # corresponding density occurrence per image sample
    for select_index, img_idx in enumerate(img_indices):
        if dataset_name == "cifar10":
            img_name = None
        else:  # imagenet
            img_name = NetworkInference.convert_imagenet_idx_to_filename(img_idx=img_idx)
            if (add_background and select_index % 2 == 0) or img_idx == -1:  # assumed 50% chance of being background
                # img_idx == -1: special handler for debugging
                img_name = "background_black.jpg"  # black
            else:
                img_name = img_name
            if read_image(img_name).shape[0] != 3:
                illegal_count += 1
                logging.warning(f"Illegal image. Input image does not have 3 RGB channels. Illegal count: "
                                f"{illegal_count}. Illegal percent: {illegal_count / img_numbers * 100:.1f}%")
                continue
        # inference
        intermediate_act = nn.extract_activation_of_an_intermediate_layer(layer_idx=layer_idx,
                                                                          img_idx=img_idx,
                                                                          img_name=img_name, )
        if isinstance(intermediate_act, torch.Tensor):
            kernels: np.ndarry = intermediate_act.numpy()
        # calculate tile-level density information
        density_list: np.ndarry
        density_occurrence: np.ndarry
        density_mean: float
        density_std: float
        if layer_idx == 0:
            enable_relu = False
        else:
            enable_relu = True
        density_list, density_occurrence, density_mean, density_std = NetworkInference.calc_density_distribution(
            op_array=intermediate_act,
            tile_size=tile_size,
            enable_relu=enable_relu)
        # calculate average tile-level density distribution
        for density_idx in range(len(density_list)):
            act_density_sample = density_list[density_idx]
            aver_density_dist[act_density_sample] += density_occurrence[density_idx] / img_numbers
        # collect average density per image
        density_mean_collect.append(density_mean)
        density_std_collect.append(density_std)  # not useful yet
        # put the sampling information in the pools
        density_list_collect.append(density_list)
        density_occurrence_collect.append(density_occurrence)
    return density_list_collect, density_occurrence_collect, aver_density_dist, \
        density_mean_collect, density_std_collect


def density_extraction(tile_size: int = 8,
                       layer_idx: int = 2,
                       img_numbers: int = 1000,
                       model_name: str = "resnet18",
                       dataset_name: str = "imagenet",
                       add_background: bool = True):
    """
    statically extract tile-level density distribution and density mean
    :param tile_size: targeted tile size
    :param layer_idx: targeted layer index to observe
    :param img_numbers: number of image samples to observe
    :param model_name: targeted inference model name, options: [resnet18, resnet50, vgg19, mobilenetv2, mobilenetv3, quant_mobilenetv2]
    :param dataset_name: targeted dataset name, options: [cifar10, imagenet]
    :param add_background: consider 50% background or not
    :return: density_list_collect: list of samples, each element containing density index of current sample
             density_occurrence_collect: list of samples, each element containing density occurrence of current sample
             aver_density_dist: dict, contains tile-level density information of the average distribution
             density_mean_collect: list of average tile-level density per sample
             density_std_collect: list of standard variance of the tile-level density per sample
    """
    # generate image indices
    if dataset_name == "cifar10":
        img_indices = np.random.randint(0, 10000, size=img_numbers)
    else:  # imagenet
        img_indices = np.random.randint(1, 40000, size=img_numbers)
    (density_list_collect, density_occurrence_collect, aver_density_dist, density_mean_collect, density_std_collect
     ) = density_extraction_with_fixed_img_indices(tile_size=tile_size,
                                                   layer_idx=layer_idx,
                                                   img_indices=img_indices,
                                                   model_name=model_name,
                                                   dataset_name=dataset_name,
                                                   add_background=add_background)
    # calc density covariance matrix
    density_covariance_matrix = density_covariance_matrix_parser(density_list_collect=density_list_collect,
                                                                 density_occurrence_collect=density_occurrence_collect)
    return density_list_collect, density_occurrence_collect, aver_density_dist, \
        density_mean_collect, density_std_collect, density_covariance_matrix


def plot_act_in_bar(tile_size: int = 8,
                    layer_idx: int = 2,
                    img_indices: list = [27411],
                    model_name: str = "resnet18",
                    dataset_name: str = "imagenet",
                    enable_extraction: bool = True,
                    add_background: bool = False):
    """
    statically plot tile-level activation density distribution (not sparsity) and density mean (not sparsity)
    :param tile_size: targeted tile size
    :param layer_idx: targeted layer index to observe
    :param img_numbers: number of image samples to observe
    :param model_name: targeted inference model name, options: [resnet18, resnet50, vgg19, mobilenetv2, mobilenetv3, quant_mobilenetv2]
    :param dataset_name: targeted dataset name, options: [cifar10, imagenet]
    :param enable_extraction: whether enable density extraction (not used)
    :param add_background: consider 50% background or not
    """
    fig, axs = plt.subplots(figsize=(5, 3), nrows=1, ncols=1)
    img_count = len(img_indices)
    # extract density information
    if enable_extraction:
        density_list_collect, density_occurrence_collect, aver_density_dist, \
            density_mean_collect, density_std_collect = density_extraction_with_fixed_img_indices(
            tile_size=tile_size,
            layer_idx=layer_idx,
            img_indices=img_indices,
            model_name=model_name,
            dataset_name=dataset_name,
            add_background=add_background)
        with open(f"tmp_act_density_results_{img_count}.pkl", "wb") as fp:
            pickle.dump((img_indices, density_list_collect, density_occurrence_collect, aver_density_dist,density_mean_collect, density_std_collect), fp)
    else:
        with open(f"tmp_act_density_results_{img_count}.pkl", "rb") as fp:
            (img_indices, density_list_collect, density_occurrence_collect, aver_density_dist, density_mean_collect, density_std_collect) = pickle.load(fp)

    # plot tile-level density distribution per image
    if len(density_list_collect) == 1:
        for i in range(len(density_list_collect)):
            x = 1 - density_list_collect[i]
            axs.bar(x, density_occurrence_collect[i], color=u'#bfe2bf', width=0.1, edgecolor='k')
            # axs.scatter(x, density_occurrence_collect[i], "--o", color='black',
            #          markerfacecolor="moccasin",
            #          markeredgecolor='black',
            #          markersize=8)
    # elif len(density_list_collect) == 2:
    #     offset = 0.2
    #     x = tile_size - np.arange(len(density_list_collect[0]))
    #     axs.bar(x / tile_size - offset / 8, density_occurrence_collect[0], color=u'#bfe2bf', width=0.05, edgecolor='k')
    #     axs.bar(x / tile_size + offset / 8, density_occurrence_collect[1], color=u'#cd87de', width=0.05, edgecolor='k')
    else:
        x = 1 - density_list_collect[0]
        x_vector = np.flip(np.arange(len(x)))  # used for plotting, np.flip is to ensure the descending order
        # to ensure np.flip is correctly used, the assertion is added  below
        assert np.all(np.sort(x)[::-1] == x), f'{x} is not in the descending order'
        y_vectors = np.array([density_occurrence_collect[i] for i in range(len(density_list_collect))])
        y_max = np.maximum.reduce(y_vectors)
        y_min = np.minimum.reduce(y_vectors)
        # Calculate statistics across y vectors
        y_mean = np.mean(y_vectors, axis=0)
        y_std = np.std(y_vectors, axis=0)
        # For 95% confidence interval, use:
        # y_error = 1.96 * y_std / np.sqrt(len(y_vectors))
        # For simple standard error of the mean (SEM):
        y_error = y_std / np.sqrt(len(y_vectors))
        normalized_three_sem = 3 * y_error
        sem_max = round(max(normalized_three_sem), 2)

        # Plot individual points with some transparency
        if len(y_vectors) <= 2:
            for y_vec in y_vectors:
                axs.plot(x, y_vec, 'o', alpha=0.2, color='gray', markersize=4)
                # axs.plot(x, y_vec, 'o--', alpha=1, color='gray', markersize=4)
        # Plot error bars
        # axs.plot(x, y_mean, 'o-', alpha=1, color='green', markersize=4)
        axs.bar(x_vector, y_mean, color=u'#bfe2bf', width=0.5, edgecolor='k')
        axs.fill_between(x_vector, y_mean + y_std, y_mean - y_std, alpha=0.3, color=u'#3498db')
        axs.errorbar(x_vector, y_mean, yerr=y_std,
                     fmt='o',  # Points connected by lines
                     color='#000000',  # Main color
                     ecolor='#34495e',  # Error bar color
                     capsize=5,
                     capthick=1,
                     elinewidth=1,
                     markersize=2,
                     label=f'±3SEM ({sem_max}) @ {img_count} Images')  # Standard Error of Mean
        axs.set_xticks(x_vector)
        axs.set_xticklabels(np.sort(x)[::-1].astype(str))
        aver_density = np.mean(density_mean_collect)
        aver_sparsity = 1 - aver_density
        std_density = np.std(density_mean_collect)
        std_sparsity = std_density
        logging.info(f"SEM (max): {sem_max}, Aver sparsity: {aver_sparsity}, 3std/mu: {3*std_sparsity/aver_sparsity}")
    # configuration
    axs.set_xlabel("ll$_{sp}$", fontsize=15)
    axs.set_ylabel("P$_{sp}$", fontsize=15)
    axs.set_ylim(bottom=0)
    # axs.grid(which="major", axis="both", color="gray", linestyle="--", linewidth=1)
    axs.set_axisbelow(True)
    # Increase tick label font size
    axs.tick_params(axis='both', which='major', labelsize=12)  # Adjust tick label size

    # axs.set_title("Sample-wise P$_{density}$ - ll$_{density}$")
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_act(tile_size: int = 8,
             layer_idx: int = 2,
             img_numbers: int = 1,
             model_name: str = "resnet18",
             dataset_name: str = "imagenet",
             enable_extraction: bool = True,
             add_background: bool = False):
    """
    statically plot tile-level activation density distribution (not sparsity) and density mean (not sparsity)
    :param tile_size: targeted tile size
    :param layer_idx: targeted layer index to observe
    :param img_numbers: number of image samples to observe
    :param model_name: targeted inference model name, options: [resnet18, resnet50, vgg19, mobilenetv2, mobilenetv3, quant_mobilenetv2]
    :param dataset_name: targeted dataset name, options: [cifar10, imagenet]
    :param enable_extraction: whether enable density extraction
    :param add_background: consider 50% background or not
    """
    fig, axs = plt.subplots(figsize=(8, 5), nrows=1, ncols=2)
    # extract density information
    if enable_extraction:
        density_list_collect, density_occurrence_collect, aver_density_dist, \
            density_mean_collect, density_std_collect, density_covariance_matrix = density_extraction(
            tile_size=tile_size,
            layer_idx=layer_idx,
            img_numbers=img_numbers,
            model_name=model_name,
            dataset_name=dataset_name,
            add_background=add_background)
    else:
        density_list_collect, density_occurrence_collect, aver_density_dist, \
            density_mean_collect, density_std_collect, density_covariance_matrix = read_pickle(
            f"./pkl/act/{dataset_name}/{model_name}/dist_{model_name}_{dataset_name}_layer{layer_idx}_tile{tile_size}.pkl")

    # plot tile-level density distribution per image
    for i in range(len(density_list_collect)):
        axs[0].plot(density_list_collect[i], density_occurrence_collect[i], "--o", color='black',
                    markerfacecolor="moccasin",
                    markeredgecolor='black',
                    markersize=8)
    # plot average tile-level density distribution
    prob_density_list = [x for x in aver_density_dist.keys()]
    density_probs = []
    for sparsity in prob_density_list:
        density_probs.append(aver_density_dist[sparsity])
    axs[0].plot(prob_density_list, density_probs, "--s", color='firebrick', markerfacecolor="firebrick",
                markeredgecolor='black', linewidth=3, markersize=8, label="Average")
    # derive mean/std across average density of all images
    density_mean_mean = round(np.mean(density_mean_collect).item(), 3)
    density_mean_std = round(np.std(density_mean_collect).item(), 3)
    # plot average density mean/std distribution
    density_mean_prob, density_mean_value = np.histogram(density_mean_collect, bins=30, density=True)
    bin_centers = (density_mean_value[:-1] + density_mean_value[1:]) / 2  # Calculate the bin centers
    axs[1].plot(bin_centers, density_mean_prob, "--o", color='black', markerfacecolor="moccasin",
                markeredgecolor='black', label='PDF (extracted)')
    # plot corresponding norm distribution
    prob_x = np.linspace(density_mean_mean - 3 * density_mean_std, density_mean_mean + 3 * density_mean_std)
    prob_y = norm.pdf(prob_x, density_mean_mean, density_mean_std)
    axs[1].plot(prob_x, prob_y, "--o", color='red', markerfacecolor=u'#ff7f0e',
                markeredgecolor='black', linewidth=3, label='PDF (Gaussian)')
    # configuration
    axs[0].set_xlabel("ll$_{density}$", fontsize=15)
    axs[0].set_ylabel("P$_{density}$", fontsize=15)
    axs[0].set_xlim([-0.05, 1.05])
    # axs[0].set_ylim([-0.05, 1.05])
    axs[1].set_xlabel("$\mu$ll$_{density}$", fontsize=15)
    axs[1].set_ylabel(f"Probability", fontsize=15)
    axs[1].set_xlim([-0.05, 1.05])
    axs[0].grid(which="major", axis="both", color="gray", linestyle="--", linewidth=1)
    axs[0].set_axisbelow(True)
    axs[0].legend(loc="upper left")
    axs[1].legend(loc="upper right")
    axs[0].set_title("Sample-wise P$_{density}$ - ll$_{density}$")
    axs[1].set_title("Average density PDF across samples")
    if enable_extraction:
        img_numbers_for_title = img_numbers
    else:
        img_numbers_for_title = len(density_list_collect)
    plt.suptitle(
        f"Model: {model_name}, dataset: {dataset_name}, Layer: {layer_idx}, #samples: {img_numbers_for_title}\n"
        f"density $\mu$: {density_mean_mean}, $\sigma$: {density_mean_std}, "
        f"3$\sigma$/$\mu$: {3 * density_mean_std / density_mean_mean * 100:.1f}%")
    plt.tight_layout()
    plt.show()


def plot_weight(tile_size: int = 8,
                layer_idx: int = 2,
                model_name: str = "resnet18",
                enable_extraction: bool = True):
    """
        statically plot tile-level activation density distribution (not sparsity) and density mean (not sparsity)
        :param tile_size: targeted tile size
        :param layer_idx: targeted layer index to observe
        :param model_name: targeted inference model name, options: [resnet18, resnet50, vgg19, mobilenetv2, mobilenetv3, quant_mobilenetv2]
        :param enable_extraction: whether enable density extraction
        """
    fig, axs = plt.subplots(figsize=(10, 3), nrows=1, ncols=2)
    # extract density information
    if enable_extraction:
        density_list, density_occurrence, density_mean, density_std = density_extraction_weight(
            tile_size=tile_size,
            layer_idx=layer_idx,
            model_name=model_name)
    else:
        density_list, density_occurrence, density_mean, density_std = read_pickle(
            f"./pkl/weight/{model_name}/dist_{model_name}_layer{layer_idx}_tile{tile_size}.pkl")

    # plot tile-level density distribution
    # axs[0].bar(density_list_collect[i], density_occurrence_collect[i], color='green', edgecolor='black', width=0.1)
    axs[0].plot(density_list, density_occurrence, "--o", color='black',
                markerfacecolor="moccasin",
                markeredgecolor="black",
                markersize=8,
                label="P$_{density}$ - ll$_{density}$ (Extracted)")
    # plot corresponding norm distribution PDF
    prob_x = np.linspace(density_mean - 3 * density_std, density_mean + 3 * density_std)
    prob_y = norm.pdf(prob_x, density_mean, density_std)
    axs[1].plot(prob_x, prob_y, "--o", color='red', markerfacecolor=u'#ff7f0e',
                markeredgecolor='black', linewidth=3, label='PDF (Gaussian)')
    # configuration
    axs[0].set_xlabel("ll$_{density}$", fontsize=15)
    axs[0].set_ylabel("P$_{density}$", fontsize=15)
    axs[0].set_xlim([-0.05, 1.05])
    # axs[0].set_ylim([-0.05, 1.05])
    axs[1].set_xlabel("ll$_{density}$", fontsize=15)
    axs[1].set_ylabel(f"Probability", fontsize=15)
    axs[1].set_xlim([-0.05, 1.05])
    axs[0].grid(which="major", axis="both", color="gray", linestyle="--", linewidth=1)
    axs[0].set_axisbelow(True)
    axs[1].legend()
    axs[0].set_title("P$_{density}$ - ll$_{density}$")
    axs[1].set_title("Approximated density PDF")
    plt.suptitle(
        f"Model: {model_name}, Layer: {layer_idx}\n"
        f"density $\mu$: {density_mean}, $\sigma$: {density_std}, "
        f"3$\sigma$/$\mu$: {3 * density_std / density_mean * 100:.1f}%")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    logging_level = logging.WARN  # logging level
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)
    ############################################
    # Global parameter setting
    tile_size = 4  # targeted tile size
    layer_idx = 2  # targeted layer
    # img_numbers = 1  # sample counts
    # img_indices = [8978]
    # img_indices = [8978, 27411]
    model_name = "resnet18"  # targeted network, options: [resnet18, resnet50, vgg19, mobilenetv2, mobilenetv3,
    # quant_mobilenetv2]
    dataset_name = "imagenet"  # targeted dataset, options: [cifar10, imagenet]
    add_background = True  # add noisy images (chance: 50%)
    enable_extraction = False  # whether or not enable runtime density info extraction
    img_indices = [random.randint(1, 40000) for _ in range(10)]
    logging.info(img_indices)
    ############################################
    plot_act_in_bar(tile_size=tile_size,
                    layer_idx=layer_idx,
                    img_indices=img_indices,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    enable_extraction=enable_extraction,
                    add_background=add_background)
    # plot_weight(tile_size=tile_size,
    #             layer_idx=layer_idx,
    #             model_name=model_name,
    #             enable_extraction=enable_extraction)
