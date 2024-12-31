import logging
import pickle
import numpy as np
from api import read_pickle, derive_model_layer_count


if __name__ == "__main__":
    """
    reporting activation density in all pkl, for a certain model-dataset
    """
    logging_level = logging.INFO  # logging level
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)
    ##############################
    ## parameters
    model_name = "resnet18"
    dataset_name = "imagenet"
    tile_size_i = 8
    ##############################
    layer_count = derive_model_layer_count(model_name)
    for layer_idx in range(layer_count):
        # read in sparsity info
        sparsity_info_file_i = f"./pkl/act/{dataset_name}/{model_name}/dist_{model_name}_{dataset_name}_layer{layer_idx}_tile{tile_size_i}.pkl"
        # read pkl
        [density_list_collect,
         density_occurrence_collect,
         aver_density_dist,
         density_mean_collect,
         density_std_collect] = read_pickle(sparsity_info_file_i)
        sample_count = len(density_mean_collect)
        # calc average tile-level density and std, 3std/density
        density = np.mean(density_mean_collect)
        std = np.std(density_mean_collect)
        logging.info(f"Model: {model_name}, layer: {layer_idx}, aver density: {density}, "
                     f"aver density std: {std}, 3std/density: {round(3 * std/density, 2)*100}%")
