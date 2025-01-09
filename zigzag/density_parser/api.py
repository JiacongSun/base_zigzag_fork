import pickle
import numpy as np
from inference_hooker import NetworkInference
from torchvision.io import read_image
import logging
import torch as torch


def save_to_pickle(obj, filename):
    with open(filename, "wb") as fp:
        pickle.dump(obj, fp)


def read_pickle(filename):
    with open(filename, "rb") as fp:
        load = pickle.load(fp)
    return load


def derive_model_layer_count(model_name: str):
    # Class NetworkInference in inference_hooker.py has its own function to derive this info
    layer_count_dict: dict = {
        "resnet18": 21,
        "resnet18_sparse": 17,  # does not have downsample layers (4) and fc layer (1)
        "resnet50": 54,  # 54 in the hooker, but 53 in zigzag onnx
        "vgg19": 19,
        "mobilenetv3": 54,
        "mobilenetv2": 53,
        "mobilenetv2_sparse": 50,  # see comments of resnet18_sparse
        "quant_mobilenetv2": 53,
        "efficientnetb0_sparse": 49,  # see comments of resnet18_sparse
    }
    return layer_count_dict[model_name]


def density_extraction_across_channel_dim(tile_size: int = 256,
                                          layer_idx: int = 17,
                                          img_indices: np.ndarray = np.random.randint(1, 10000, size=1),
                                          model_name: str = "resnet18",
                                          dataset_name: str = "imagenet"):
    """
    statically extract channel-wise activation density, with fixed img indices
    :param tile_size: targeted tile size
    :param layer_idx: targeted layer index to observe
    :param img_indices: image samples to observe
    :param model_name: targeted inference model name, options: [resnet18, resnet50, vgg19, mobilenetv2, mobilenetv3, quant_mobilenetv2]
    :param dataset_name: targeted dataset name, options: [cifar10, imagenet]
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
    density_collect = []
    # initialize average distribution: dict
    aver_density_dist = {i / tile_size: 0 for i in range(0, tile_size + 1)}
    # iterate across image indices
    illegal_count = 0
    # create pools for density information per sample
    tensor_collect = []
    density_mean_collect = []
    density_std_collect = []
    density_list_collect = []  # density list per image sample
    density_occurrence_collect = []  # corresponding density occurrence per image sample
    for img_idx in img_indices:
        if dataset_name == "cifar10":
            img_name = None
        else:  # imagenet
            img_name = NetworkInference.convert_imagenet_idx_to_filename(img_idx=img_idx)
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
        else:
            kernels: np.ndarry = intermediate_act
        tensor_collect.append(kernels)
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
    return tensor_collect
