import logging
import pickle
import numpy as np
from api import read_pickle, derive_model_layer_count


if __name__ == "__main__":
    """
    reporting weight density, for a certain model-dataset
    """
    logging_level = logging.INFO  # logging level
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)
    ##############################
    ## parameters
    model_name = "vgg19"
    tile_size = 8
    ##############################
    layer_count = derive_model_layer_count(model_name)
    for layer_idx in range(layer_count):
        # read in sparsity info
        sparsity_info_file_w = f"./pkl/weight/{model_name}/dist_{model_name}_layer{layer_idx}_tile{tile_size}.pkl"
        # read pkl
        [density_list, density_occurrence, density_mean, density_std] = read_pickle(sparsity_info_file_w)
        logging.info(f"Model: {model_name}, layer: {layer_idx}, aver density: {density_mean}, "
                     f"aver density std: 0")
