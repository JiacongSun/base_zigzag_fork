import os
import matplotlib.pyplot as plt
from visualization_act_density_static import density_extraction_weight
import numpy as np
from scipy.stats import norm
from torchvision.io import read_image
import logging
import pickle
import time
from tqdm import tqdm
from inference_hooker import NetworkInference
import torch as torch
from api import save_to_pickle, read_pickle, derive_model_layer_count

"""
This script will save the following information to pkl, for each layer:
- a): the layer shape
"""


def extract_laye_shape(model_name: str = "resnet18", ):
    # use models inherited from pytorch
    # initialize the model
    nn = NetworkInference(model_name=model_name)
    # sparsity parsing
    layer_shape_collect: list = nn.extract_layer_shape(return_all_layers=True)
    return layer_shape_collect


def save_layer_shape_information_in_pkl(
        model_name: str = "resnet18",
):
    START_TIME = time.time()
    # extract density information
    layer_shape_collect: list = extract_laye_shape(model_name=model_name, )
    # create output folder
    model_name_pkl = model_name
    folder_path = f"./pkl/layer_shape/{model_name_pkl}"
    os.makedirs(folder_path, exist_ok=True)
    # save information to pkl
    information_to_be_saved: list = layer_shape_collect
    save_to_pickle(obj=information_to_be_saved,
                   filename=f"{folder_path}/shape_{model_name_pkl}.pkl")
    # timing report
    END_TIME = time.time()
    time_in_second = round(END_TIME - START_TIME, 2)
    logging.debug(f"Total extraction time (seconds): {time_in_second}")


if __name__ == "__main__":
    model_name = "mobilenetv2"  # targeted model name, [resnet18, resnet50, vgg19, mobilenetv2, mobilenetv3, quant_mobilenetv2, resnet18_sparse, "mobilenetv2_sparse", "efficientnetb0_sparse"]
    save_layer_shape_information_in_pkl(
        model_name=model_name,
    )
