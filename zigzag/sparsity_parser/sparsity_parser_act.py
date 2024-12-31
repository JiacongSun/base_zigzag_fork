import os
import matplotlib.pyplot as plt
from visualization_static import density_extraction_with_fixed_img_indices
import numpy as np
from scipy.stats import norm
from torchvision.io import read_image
import logging
import pickle
import time
from tqdm import tqdm
from api import save_to_pickle, read_pickle, derive_model_layer_count

"""
This script will save the following information to pkl, for each layer and each image:
- a): list of tile-level activation density (not sparsity) levels
- b): list of tile-level activation density (not sparsity) occurrences
- c): average tile-level activation density distribution
- d): the average activation density
- e): the activation density std across tiles
"""


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


def save_act_sparsity_information_in_pkl(
        tile_size: int = 8,
        layer_idx: int = 2,
        img_indices: np.ndarray = np.random.randint(1, 10000, size=1),
        model_name: str = "resnet18",
        dataset_name: str = "imagenet"):
    """
    statically plot tile-level density distribution and density mean
    :param tile_size: targeted tile size
    :param layer_idx: targeted layer index to observe
    :param img_indices: image samples to observe
    :param model_name: targeted inference model name, options: [resnet18, resnet50, vgg19, mobilenetv2, mobilenetv3, quant_mobilenetv2]
    :param dataset_name: targeted dataset name, options: [cifar10, imagenet]
    """
    START_TIME = time.time()
    # extract density information
    density_list_collect: list
    density_occurrence_collect: list
    aver_density_dist: dict
    density_mean_collect: list
    density_std_collect: list
    density_covariance_matrix: np.ndarray
    density_list_collect, density_occurrence_collect, aver_density_dist, \
        density_mean_collect, density_std_collect = density_extraction_with_fixed_img_indices(tile_size=tile_size,
                                                                                              layer_idx=layer_idx,
                                                                                              img_indices=img_indices,
                                                                                              model_name=model_name,
                                                                                              dataset_name=dataset_name)
    # calc density covariance matrix
    density_covariance_matrix = density_covariance_matrix_parser(density_list_collect=density_list_collect,
                                                                 density_occurrence_collect=density_occurrence_collect)

    # create output folder
    folder_path = f"./pkl/act/{dataset_name}/{model_name}"
    os.makedirs(folder_path, exist_ok=True)
    # save information to pkl
    information_to_be_saved: list = [density_list_collect,
                                     density_occurrence_collect,
                                     aver_density_dist,
                                     density_mean_collect,
                                     density_std_collect,
                                     density_covariance_matrix]
    save_to_pickle(obj=information_to_be_saved,
                   filename=f"{folder_path}/dist_{model_name}_{dataset_name}_layer{layer_idx}_tile{tile_size}.pkl")
    # timing report
    END_TIME = time.time()
    time_in_second = round(END_TIME - START_TIME, 2)
    logging.debug(f"Total extraction time (seconds): {time_in_second}")


if __name__ == "__main__":
    logging_level = logging.WARN  # logging level
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)
    ######################################################
    ## parameters
    dataset_name = "imagenet"  # targeted dataset
    model_name = "resnet18"  # targeted model name
    img_numbers = 1000  # number of img samples
    tile_size = 8  # targeted tile size
    ######################################################
    # generate image indices
    if dataset_name == "cifar10":
        img_indices = np.random.randint(0, 10000, size=img_numbers)
    else:  # imagenet
        img_indices = np.random.randint(1, 40000, size=img_numbers)
    # model layer count extraction
    layer_count = derive_model_layer_count(model_name=model_name)

    progress_bar = tqdm(total=layer_count, desc=f"{dataset_name}, {model_name}, layer")
    for layer_idx in range(layer_count):
        save_act_sparsity_information_in_pkl(
            tile_size=tile_size,
            layer_idx=layer_idx,
            img_indices=img_indices,
            model_name=model_name,
            dataset_name=dataset_name
        )
        progress_bar.update(1)
