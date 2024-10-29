import os
import sys

sys.path.insert(0, os.path.dirname(os.getcwd()))
from zigzag.api import get_hardware_performance_zigzag
from zigzag.hardware.architecture.get_cacti_cost import get_cacti_cost
import yaml
import logging
import time
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd
import re
import pickle
from collections import OrderedDict
import numpy as np


def normalize_vector(vec):
    """:param vec: input numpy vector"""
    min_value = np.min(vec)
    assert min_value > 0
    norm_vec = vec / min_value
    return norm_vec


def plot_across_varied_foms(df: pd.DataFrame, carbon_parameters: dict):
    """
    Experiment 1: different FoMs lead to different architectures.
    :param df: source data
    :param carbon_parameters: carbon parameters
    """
    # extract carbon parameters
    ci_op = carbon_parameters["ci_op"]  # g,CO2/pJ
    ci_em = carbon_parameters["ci_em"]  # g,CO2/mm2@28nm
    chip_yield = carbon_parameters["chip_yield"]
    lifetime = carbon_parameters["lifetime"]  # unit: ns
    rci = ci_op * lifetime / (ci_em / chip_yield)  # ns * mm2
    # filter df
    df_cc = df[((df.d1*df.d2*df.d3 == 1024) & (df.mem_size == 256*1024*8)) | ((df.d1*df.d2*df.d3 == 16*1024) & (df.mem_size == 1024*1024*8))]
    ############################################
    ## data preprocessing
    # calc carbon
    delay = df_cc["cycles"] * df_cc["tclk"]  # ns
    area = df_cc["area"]  # mm2
    energy = df_cc["energy"]  # pJ
    c_op = energy * ci_op
    c_em = delay * area * ci_em / (chip_yield * lifetime)
    c_tot = c_op + c_em
    # calc #PEs
    pe_num = list(df_cc["d1"] * df_cc["d2"] * df_cc["d3"] // 1024)  # unit: k
    mem_size = list(df_cc["mem_size"] // 1024 // 8)  # unit: kB
    voltage = list(df_cc["voltage"])
    x_ticklabels = [f"{pe_num[idx]}k, {mem_size[idx]}kB@{voltage[idx]}V" for idx in range(len(pe_num))]
    ############################################
    ## data calculation
    # TODO: calc metric 1: EDP
    metric_edp = energy * delay  # pJ * ns
    metric_edp = normalize_vector(metric_edp)
    # TODO: calc metric 2: CDP
    metric_cdp = c_tot * delay  # g,CO2 * ns
    metric_cdp = normalize_vector(metric_cdp)
    # TODO: calc metric 3: CEP
    metric_cep = c_tot * energy  # g,CO2 * pJ
    metric_cep = normalize_vector(metric_cep)
    # TODO: calc metric 4: C2EP
    metric_c2ep = c_tot * c_tot * energy  # g,CO2^2 * pJ
    metric_c2ep = normalize_vector(metric_c2ep)
    # TODO: calc metric 5: CE2P
    metric_ce2p = c_tot * energy * energy  # g,CO2 * pJ^2
    metric_ce2p = normalize_vector(metric_ce2p)
    # TODO: calc metric 6: tCDP
    metric_tcdp = c_em * delay  # g,CO2 * ns
    metric_tcdp = normalize_vector(metric_tcdp)
    # TODO: calc metric 7: CNADP
    metric_cnadp = area * delay + (energy * rci)
    metric_cnadp = normalize_vector(metric_cnadp)
    # TODO: calc metric 8: normalized c_tot
    c_tot = normalize_vector(c_tot)
    # TODO: calc metric e, d, a
    energy = normalize_vector(energy)
    delay = normalize_vector(delay)
    area = normalize_vector(area)
    ############################################
    ## TODO: plot
    ################
    # plot setting
    bar_width = 0.1
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ################
    fig, ax = plt.subplots(figsize=(5, 3))
    index = np.arange(len(metric_cnadp))
    # metrics = [delay, area, energy, metric_edp, metric_cdp, metric_cep, metric_c2ep, metric_ce2p, metric_tcdp, metric_cnadp, c_tot]
    # labels = ["Delay", "Area", "Energy", "EDP", "CDP", "CEP", "C2EP", "CE2P", "tCDP", "CNADP", "C"]
    metrics = [metric_edp, metric_cdp, metric_cep, metric_c2ep, metric_ce2p, metric_tcdp, metric_cnadp, c_tot]
    labels = ["EDP", "CDP", "CEP", "C2EP", "CE2P", "tCDP", "CNADP", r"C$_{tot}$"]
    for i, vec in enumerate(metrics):
        logging.info(f"Index {i} of min: {np.argmin(vec)}")
    for i, vec in enumerate(metrics):
        ax.bar(index + i * bar_width, vec, edgecolor="black", width=bar_width, label=f"{labels[i]}", color=colors[i%len(colors)])
        min_idx = np.argmin(vec)
        min_value = np.min(vec)
        ax.plot(min_idx + i * bar_width, min_value, "--*", markeredgecolor="black",
                markersize=6, markerfacecolor=colors[i%len(colors)])
        ## in curve
        # if i == len(metrics) - 1:
        #     plt.plot(index, vec, label=f"{labels[i]}", linestyle=":", color="black", linewidth=2)
        # else:
        #     plt.plot(index, vec, label=f"{labels[i]}")
    # TODO: change x tick labels
    ax.set_xticks(index + len(metrics)//2 * bar_width)
    ax.set_xticklabels(x_ticklabels,
                       rotation=30,
                       multialignment="center",
                       fontsize=10,
                       fontweight="bold")
    ax.set_yticks(np.arange(0, 10, 1))

    # ax.set_xlabel("HW cases")
    ax.set_ylabel("Normalized metrics", fontsize=12, fontweight="bold")
    # ax.set_yscale("log")
    ax.grid(visible=True, which="both", axis="y", linestyle="--", color="black")
    ax.set_axisbelow(True)

    plt.legend(ncol=4, loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """
    Experiments visualization
    """
    logging_level = logging.INFO  # logging level
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)
    ###################################################
    ## file setting
    output_folder = "./pkl/"
    pkl_name_list = ["resnet8_1.0.pkl", "resnet8_0.8.pkl", "resnet8_0.6.pkl"]
    ###################################################
    ## carbon setting
    carbon_parameters = {
        "ci_em": 14.13,  # g,CO2/mm2@28nm
        "ci_op": 24 / 3.6e+18,  # g,CO2/pJ
        "chip_yield": 0.95,
        "lifetime": 94608000e+9,  # 3 years in ns
    }
    ###################################################
    ans = []
    for output_pkl_name in pkl_name_list:
        with open(output_folder+output_pkl_name, "rb") as fp:
            ans.append(pickle.load(fp))
    ans_tot = pd.concat(ans)
    plot_across_varied_foms(df=ans_tot, carbon_parameters=carbon_parameters)
