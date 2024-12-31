import os
import matplotlib.pyplot as plt
from visualization_static import density_extraction_weight
import numpy as np
from scipy.stats import norm
from torchvision.io import read_image
import logging
import pickle
import time
from tqdm import tqdm
from api import save_to_pickle, read_pickle, derive_model_layer_count

"""
This script will save the following information to pkl, for each layer:
- a): list of tile-level weight density (not sparsity) levels
- b): list of tile-level weight density (not sparsity) occurrences
- c): the average weight density
- d): the weight density std across tiles
"""


def save_weight_sparsity_information_in_pkl(
        tile_size: int = 8,
        layer_idx: int = 2,
        model_name: str = "resnet18", ):
    """
    statically plot tile-level density distribution and density mean
    :param tile_size: targeted tile size
    :param layer_idx: targeted layer index to observe
    :param model_name: targeted inference model name, options: [resnet18, resnet50, vgg19, mobilenetv2, mobilenetv3, quant_mobilenetv2]
    """
    START_TIME = time.time()
    # extract density information
    density_list: np.ndarry
    density_occurrence: np.ndarry
    density_mean: float
    density_std: float
    density_list, density_occurrence, density_mean, density_std = density_extraction_weight(tile_size=tile_size,
                                                                                            layer_idx=layer_idx,
                                                                                            model_name=model_name, )
    # create output folder
    if model_name == "resnet18_sparse":  # replace with sparse resnet18
        model_name_pkl = "resnet18"
    else:
        model_name_pkl = model_name
    folder_path = f"./pkl/weight/{model_name_pkl}"
    os.makedirs(folder_path, exist_ok=True)
    # save information to pkl
    information_to_be_saved: list = [density_list,
                                     density_occurrence,
                                     density_mean,
                                     density_std]
    save_to_pickle(obj=information_to_be_saved,
                   filename=f"{folder_path}/dist_{model_name_pkl}_layer{layer_idx}_tile{tile_size}.pkl")
    # timing report
    END_TIME = time.time()
    time_in_second = round(END_TIME - START_TIME, 2)
    logging.debug(f"Total extraction time (seconds): {time_in_second}")


if __name__ == "__main__":
    model_name = "resnet18_sparse"
    tile_size = 8
    layer_count = derive_model_layer_count(model_name=model_name)
    progress_bar = tqdm(total=layer_count, desc=f"{model_name}, layer")
    for layer_idx in range(layer_count):
        save_weight_sparsity_information_in_pkl(
            model_name=model_name,
            tile_size=tile_size,
            layer_idx=layer_idx,
        )
        progress_bar.update(1)
