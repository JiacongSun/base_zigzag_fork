import pickle
import time
import sys
import os
import logging

# append the parent folder to the environment
sys.path.insert(0, os.path.dirname(os.getcwd()))

import math
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
from tqdm import tqdm
import plotly.express as px
import pandas as pd
from scipy.stats import norm
from zigzag.classes.stages.SparseObject import SparseObject
from zigzag.classes.cost_model.sparse_cost_model_assist import Distribution
from validation_datapath import ValidationDatapath, read_pickle
from validation_memory import joint_covariance_extraction, sparsity_std_extraction
from create_dstc_core import create_dstc_core
from create_core import create_core  # basic naive template
from zigzag.classes.stages.SparseObject import SparseObject
from zigzag.classes.cost_model.sparse_cost_model_assist import Distribution
from zigzag.classes.workload.layer_node import LayerNode
from zigzag.classes.mapping.temporal.temporal_mapping import TemporalMapping
from zigzag.classes.stages import *


if __name__ == "__main__":
    """
    system validation experiments
    Workload: ResNet18, L2
    Dataset: Imagenet
    Hardware: SIGMA-like (8 x 8 PE array)
    Sparsity: random activation sparsity, extracted weight sparsity
    Technique: gating on activation, skipping on weight
    Note: the sparsity distribution .pkl file need to be extracted separately by script
        ./density_parser/density_parser_act.py
    """
    logging_level = logging.CRITICAL  # logging level
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)
    #############################################
    ## parameters
    sparsity_variation_type = "iteru"
    model_name = "resnet18"
    dataset_name = "cifar10"
    tile_size = 8
    fix_temporal_mapping = False
    #############################################
    workload_filename = get_workload_filename(model_name=model_name)
    dut = ValidationSystem(fix_temporal_mapping=fix_temporal_mapping, workload_filename=workload_filename)
    layer_count: int = get_layer_count_in_model(model_name=model_name)
    time_sum_sparse = 0
    time_sum_sample = 0
    time_sum_pdf = 0
    for targeted_layer_idx in range(0, layer_count):
        # for targeted_layer_idx in range(18, 19):
        logging.critical(f"Processing {dataset_name}, {model_name}, layer: {targeted_layer_idx}")
        # dut.validation_system(model_name=model_name,
        #                       dataset_name=dataset_name,
        #                       tile_size=tile_size,
        #                       variation_type=sparsity_variation_type,
        #                       targeted_layer_idx=targeted_layer_idx)  # simulation and save results to pkl
        # dut.plot(model_name=model_name,
        #          dataset_name=dataset_name,
        #          variation_type=sparsity_variation_type,
        #          targeted_layer_idx=targeted_layer_idx,
        #          plot_simulation_time=False)  # read pkl and plotting
        time_number, time_sample, time_pdf = dut.sim_time_comparison(model_name=model_name,
                                dataset_name=dataset_name,
                                variation_type=sparsity_variation_type,
                                targeted_layer_idx=targeted_layer_idx)  # read pkl and plotting
        time_sum_sparse += time_number
        time_sum_sample += time_sample
        time_sum_pdf += time_pdf
        print("\n")
    print(f"Time (s): [{model_name}@{dataset_name}] [Number]: {time_sum_sparse}, [Sample]: {time_sum_sample}, [PDF]: {time_sum_pdf}")
