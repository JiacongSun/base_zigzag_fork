import os
import pickle
import logging
from visualization_act_density_static import density_covariance_matrix_parser
from api import read_pickle, derive_model_layer_count, save_to_pickle
import numpy as np

if __name__ == "__main__":
    logging_level = logging.WARN  # logging level
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)
    ######################################################
    ## parameters
    dataset_name = "cifar10"  # targeted dataset, [cifar10, imagenet]
    model_name = "resnet50"  # targeted model name, [resnet18, resnet50, vgg19, mobilenetv2, mobilenetv3, quant_mobilenetv2]
    tile_size_i = 8  # targeted tile size
    ######################################################

    layer_count = derive_model_layer_count(model_name)
    layer_list_collect_across_samples = []
    density_mean_collect_across_samples = []
    density_mean = []
    for layer_idx in range(layer_count):
        # read in sparsity info
        sparsity_info_file_i = f"./pkl/act/{dataset_name}/{model_name}/dist_{model_name}_{dataset_name}_layer{layer_idx}_tile{tile_size_i}.pkl"
        # read pkl
        [density_list_collect,
         density_occurrence_collect,
         aver_density_dist,
         density_mean_collect,
         density_std_collect,
         density_covariance_matrix] = read_pickle(sparsity_info_file_i)
        sample_count = len(density_mean_collect)
        # calc average tile-level density
        density = np.mean(density_mean_collect)
        # append results
        layer_list_collect_across_samples = [np.array([1] * layer_count)] * sample_count
        if len(density_mean_collect_across_samples) != sample_count:
            for density_sample in density_mean_collect:
                density_mean_collect_across_samples.append([density_sample])
        else:
            for density_sample_idx in range(sample_count):
                density_sample = density_mean_collect[density_sample_idx]
                density_mean_collect_across_samples[density_sample_idx].append(density_sample)
    density_mean_collect_across_samples = [np.array(ele) for ele in density_mean_collect_across_samples]
    density_covariance_matrix = density_covariance_matrix_parser(density_list_collect=layer_list_collect_across_samples,
                                                                 density_occurrence_collect=density_mean_collect_across_samples)
    os.makedirs(f"pkl/layerwise_cm/{dataset_name}/", exist_ok=True)
    save_to_pickle(density_covariance_matrix, f"pkl/layerwise_cm/{dataset_name}/cm_{model_name}_tile{tile_size_i}.pkl")
    pass
