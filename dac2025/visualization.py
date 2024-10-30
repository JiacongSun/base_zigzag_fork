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
import plotly.express as px


def save_as_pickle(df, filename):
    with open(filename, "wb") as fp:
        pickle.dump(df, fp)


def normalize_vector(vec, ref=None):
    """:param vec: input numpy vector"""
    if ref is None:
        min_value = np.min(vec)
    else:
        min_value = ref
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
    df_cc = df[((df.d1 * df.d2 * df.d3 == 1024) & (df.mem_size == 256 * 1024 * 8)) | (
            (df.d1 * df.d2 * df.d3 == 16 * 1024) & (df.mem_size == 1024 * 1024 * 8))]
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
    # calc metric 1: EDP
    metric_edp = energy * delay  # pJ * ns
    metric_edp = normalize_vector(metric_edp)
    # calc metric 2: CDP
    metric_cdp = c_tot * delay  # g,CO2 * ns
    metric_cdp = normalize_vector(metric_cdp)
    # calc metric 3: CEP
    metric_cep = c_tot * energy  # g,CO2 * pJ
    metric_cep = normalize_vector(metric_cep)
    # calc metric 4: C2EP
    metric_c2ep = c_tot * c_tot * energy  # g,CO2^2 * pJ
    metric_c2ep = normalize_vector(metric_c2ep)
    # calc metric 5: CE2P
    metric_ce2p = c_tot * energy * energy  # g,CO2 * pJ^2
    metric_ce2p = normalize_vector(metric_ce2p)
    # calc metric 6: tCDP
    metric_tcdp = c_em * delay  # g,CO2 * ns
    metric_tcdp = normalize_vector(metric_tcdp)
    # calc metric 7: CNADP
    metric_cnadp = area * delay + (energy * rci)
    metric_cnadp = normalize_vector(metric_cnadp)
    # calc metric 8: normalized c_tot
    c_tot = normalize_vector(c_tot)
    # calc metric e, d, a
    energy = normalize_vector(energy)
    delay = normalize_vector(delay)
    area = normalize_vector(area)
    ############################################
    ## plot
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
        ax.bar(index + i * bar_width, vec, edgecolor="black", width=bar_width, label=f"{labels[i]}",
               color=colors[i % len(colors)])
        min_idx = np.argmin(vec)
        min_value = np.min(vec)
        ax.plot(min_idx + i * bar_width, min_value, "--*", markeredgecolor="black",
                markersize=6, markerfacecolor=colors[i % len(colors)])
    # change x tick labels
    ax.set_xticks(index + len(metrics) // 2 * bar_width)
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


def plot_across_varied_carbon_source(df: pd.DataFrame,
                                     carbon_parameters: dict,
                                     interactive_plot: bool
                                     ):
    """
    Experiment 2: impacts of varied power source or embodied carbon intensity
    :param df: source data
    :param carbon_parameters: carbon parameters
    :param interactive_plot: True: show plotting in plotly.express
    """
    # extract carbon parameters
    ci_op_candidates = np.array([820, 230, 11]) / 3.6e+18  # g,CO2/pJ
    # ci_em_candidates = np.array([14.13, 11.16, 8.82, 7.12, 7.1, 7, 6.9, 6.85])  # g,CO2/mm2@28nm
    ci_em_candidates = np.array([14.13])

    chip_yield = carbon_parameters["chip_yield"]
    lifetime = carbon_parameters["lifetime"]  # unit: ns

    # filter df
    # df_cc = df[((df.d1*df.d2*df.d3 == 1024) & (df.mem_size == 256*1024*8)) | ((df.d1*df.d2*df.d3 == 16*1024) & (df.mem_size == 1024*1024*8))]
    df_cc = df
    ############################################
    ## data preprocessing
    # calc carbon
    delay = df_cc["cycles"] * df_cc["tclk"]  # ns
    area = df_cc["area"]  # mm2
    energy = df_cc["energy"]  # pJ
    adp = delay * area
    # calc #PEs
    pe_num = list(df_cc["d1"] * df_cc["d2"] * df_cc["d3"] // 1024)  # unit: k
    mem_size = list(df_cc["mem_size"] // 1024 // 8)  # unit: kB
    voltage = list(df_cc["voltage"])
    x_ticklabels = [f"{pe_num[idx]}k, {mem_size[idx]}kB@{voltage[idx]}V" for idx in range(len(pe_num))]
    ############################################
    ## plot
    ################
    ## interactive plotting
    if interactive_plot:
        dft = pd.DataFrame(
            {"e": energy, "adp": adp, "info": x_ticklabels, "d1": df.d1, "d2": df.d2, "d3": df.d3, "area": area})
        fig = px.scatter(dft, x="e", y="adp", hover_data=["info", "d1", "d2", "d3", "area"])
        fig.show()
    ################
    # plot setting
    slides = 1  # nb of outlines to plot
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ################
    fig, ax = plt.subplots(figsize=(5, 3))
    adp = list(adp)
    energy = list(energy)
    for i in range(len(adp)):
        ax.scatter(energy[i], adp[i], s=30, marker="o", edgecolor="black", label=x_ticklabels[i])
    # plot outline for CNADP
    for ci_op_idx in range(len(ci_op_candidates)):
        for ci_em_idx in range(len(ci_em_candidates)):
            ci_op = ci_op_candidates[ci_op_idx]
            ci_em = ci_em_candidates[ci_em_idx]
            color_idx = (ci_op_idx * len(ci_em_candidates) + ci_em_idx) % len(colors)
            rci = ci_op * lifetime / (ci_em / chip_yield)  # ns * mm2 / pJ
            print(rci)
            adp_min = min(adp)
            adp_max = max(adp)
            energy_min = max(min(energy), adp_min / rci)
            energy_max = max(max(energy), adp_max / rci)

            related_adp_min = energy_min * rci
            related_adp_max = energy_max * rci
            ax.plot([0, energy_min], [related_adp_min, 0], linestyle="--", color=colors[color_idx])
            # ax.plot([0, energy_max], [related_adp_max, 0], linestyle="--", color=colors[color_idx])
            # for i in range(1, slides):
            #     energy_curr = energy_min + (energy_max - energy_min) * i / slides
            #     related_adp = energy_curr * rci
            #     ax.plot([0, energy_curr], [related_adp, 0], linestyle="--", color=colors[color_idx])

    ax.set_xlabel("Energy [pJ]", fontsize=12, fontweight="bold")
    ax.set_ylabel(u"ADP [mm$^2$$\cdot$ns]", fontsize=12, fontweight="bold")
    ax.set_xlim([0, max(energy) * 1.1])
    # ax.set_ylim([0, max(adp)*1.1])
    adp_second_max = sorted(set(adp), reverse=True)[1]
    ax.set_ylim([0, adp_second_max * 1.1])  # calibration for this experiment
    # ax.grid(visible=True, which="both", axis="y", linestyle="--", color="black")
    # ax.set_axisbelow(True)

    # plt.legend(ncol=4, loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_on_algorithm_impact(df: pd.DataFrame,
                             carbon_parameters: dict,
                             ):
    """
    plot the impact of number of #ops
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
    df_cc = df[((df.d1 * df.d2 * df.d3 == 1024) & (df.mem_size == 256 * 1024 * 8)) | (
            (df.d1 * df.d2 * df.d3 == 16 * 1024) & (df.mem_size == 1024 * 1024 * 8))]
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
    # calc metric: CNADP
    metric_cnadp = area * delay + (energy * rci)

    # calc embodied cost
    metric_cnadp_em = area * delay
    metric_cnadp_em = normalize_vector(metric_cnadp_em, ref=min(metric_cnadp))
    # calc operational cost
    metric_cnadp_op = energy * rci
    metric_cnadp_op = normalize_vector(metric_cnadp_op, ref=min(metric_cnadp))

    metric_cnadp = normalize_vector(metric_cnadp)
    ############################################
    ## plot
    ################
    # plot setting
    bar_width = 0.1
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ################
    # TODO: construct fake data temporarily
    metric_cnadp_size_double = metric_cnadp * 2
    metric_cnadp_size_trible = metric_cnadp * 3
    metric_cnadp_em_size_double = metric_cnadp_em * 2
    metric_cnadp_em_size_trible = metric_cnadp_em * 3
    metric_cnadp_op_size_double = metric_cnadp_op * 2
    metric_cnadp_op_size_trible = metric_cnadp_op * 3
    # TODO: finish

    fig, ax = plt.subplots(figsize=(5, 3))
    index = np.arange(len(metric_cnadp))
    labels = ["3Resnet8", "2Resnet8", "Resnet8"]
    metrics = [metric_cnadp_size_trible, metric_cnadp_size_double, metric_cnadp]
    metrics_em = [metric_cnadp_em_size_trible, metric_cnadp_em_size_double, metric_cnadp_em]
    metrics_op = [metric_cnadp_op_size_trible, metric_cnadp_op_size_double, metric_cnadp_op]
    hatchs = ["//", "--", "\\\\"]
    for i, vec in enumerate(metrics):
        ax.bar(index + i * bar_width, metrics_em[i], edgecolor="black", width=bar_width, label=f"{labels[i]}",
               color=colors[0], hatch=hatchs[i])
        ax.bar(index + i * bar_width, metrics_op[i], edgecolor="black", width=bar_width,
               bottom=metrics_em[i],
               color=colors[1], hatch=hatchs[i])
    # change x tick labels
    ax.set_xticks(index + len(metrics) // 2 * bar_width)
    ax.set_xticklabels(x_ticklabels,
                       rotation=30,
                       multialignment="center",
                       fontsize=10,
                       fontweight="bold")
    ax.set_yticks(np.arange(0, max(metrics[0]) + 1, 1))

    ax.set_ylabel("Normalized CNADP", fontsize=12, fontweight="bold")
    ax.grid(visible=True, which="both", axis="y", linestyle="--", color="black")
    ax.set_axisbelow(True)

    plt.legend(ncol=3, loc="upper center")
    plt.tight_layout()
    plt.show()


def plot_lifetime_impact(df: pd.DataFrame,
                         carbon_parameters: dict,
                         ):
    """
    plot lifetime impact
    """
    ############################################
    ## setting
    max_nb_apps = 8  # number of workloads to support on the same hardware
    ############################################
    # extract carbon parameters
    ci_op = carbon_parameters["ci_op"]  # g,CO2/pJ
    ci_em = carbon_parameters["ci_em"]  # g,CO2/mm2@28nm
    chip_yield = carbon_parameters["chip_yield"]
    lifetime = carbon_parameters["lifetime"]  # unit: ns
    rci = ci_op * lifetime / (ci_em / chip_yield)  # ns * mm2
    # filter df
    # df_cc = df[((df.d1 * df.d2 * df.d3 == 1024) & (df.mem_size == 256 * 1024 * 8)) | (
    #         (df.d1 * df.d2 * df.d3 == 16 * 1024) & (df.mem_size == 1024 * 1024 * 8))]
    df_cc = df
    ############################################
    ## for figure a
    ############################################
    ## data preprocessing
    target = df.iloc[0]
    # calc carbon
    delay = target["cycles"] * target["tclk"]  # ns
    area = target["area"]  # mm2
    energy = target["energy"]  # pJ
    c_op = energy * ci_op
    c_em = area * delay * ci_em / (chip_yield * lifetime)
    c_tot = c_op + c_em  # per inference
    rci = ci_op * lifetime / (ci_em / chip_yield)  # ns * mm2
    # calc metric: CNADP
    metric_cnadp_em = area * delay
    # collect data
    norm_cnadp_em_collect = []
    norm_cnadp_op_collect = []
    carbon_em_collect = []
    carbon_op_collect = []
    carbon_tot_collect = []
    for nb_app in range(1, max_nb_apps+1):
        rci = ci_op * nb_app * lifetime / (ci_em / chip_yield)  # ns * mm2
        metric_cnadp_op = energy * rci
        metric_cnadp = metric_cnadp_em + metric_cnadp_op
        norm_metric_cnadp_em = metric_cnadp_em / metric_cnadp
        norm_metric_cnadp_op = metric_cnadp_op / metric_cnadp
        norm_cnadp_em_collect.append(norm_metric_cnadp_em)
        norm_cnadp_op_collect.append(norm_metric_cnadp_op)
        # TODO: debugging - calc carbon/inf
        carbon_em = area * delay * ci_em / (chip_yield * nb_app * lifetime)
        carbon_op = energy * ci_op
        carbon_tot = carbon_em + carbon_op
        carbon_em_collect.append(carbon_em)
        carbon_op_collect.append(carbon_op)
        carbon_tot_collect.append(carbon_tot)
        # TODO: finish
    norm_cnadp_em_collect = np.array(norm_cnadp_em_collect)
    norm_cnadp_op_collect = np.array(norm_cnadp_op_collect)
    # TODO: debugging - calc carbon/inf
    carbon_em_collect = np.array(carbon_em_collect)
    carbon_op_collect = np.array(carbon_op_collect)
    carbon_tot_collect = np.array(carbon_tot_collect)
    carbon_em_collect = normalize_vector(carbon_em_collect, ref=max(carbon_tot_collect))
    carbon_op_collect = normalize_vector(carbon_op_collect, ref=max(carbon_tot_collect))
    carbon_tot_collect = normalize_vector(carbon_tot_collect, ref=max(carbon_tot_collect))
    # TODO: finish

    # calc #PEs for labeling
    pe_num = list(df_cc["d1"] * df_cc["d2"] * df_cc["d3"] // 1024)  # unit: k
    mem_size = list(df_cc["mem_size"] // 1024 // 8)  # unit: kB
    voltage = list(df_cc["voltage"])
    pe_ticklabels = [f"{pe_num[idx]}k, {mem_size[idx]}kB@{voltage[idx]}V" for idx in range(len(pe_num))]
    x_ticklabels = [f"{i+1}" for i in range(max_nb_apps)]

    ############################################
    ## plot
    ################
    # plot setting
    bar_width = 0.5
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ################
    fig, ax = plt.subplots(ncols=2, figsize=(10, 3))
    index = np.arange(len(norm_cnadp_em_collect))
    ax[0].bar(index, norm_cnadp_em_collect, edgecolor="black", width=bar_width,
              color=colors[0])
    ax[0].bar(index, norm_cnadp_op_collect, edgecolor="black", width=bar_width,
              bottom=norm_cnadp_em_collect,
              color=colors[1])
    ax[0].plot(index, carbon_tot_collect, "--*", markersize=10,
               markerfacecolor="white", markeredgecolor="black", color="black", linewidth=2)
    # change x tick labels
    ax[0].set_xticks(index)
    ax[0].set_xticklabels(x_ticklabels,
                       rotation=0,
                       multialignment="center",
                       fontsize=10,
                       fontweight="normal")

    ax[0].set_ylabel("Normalized CNADP", fontsize=12, fontweight="bold")
    ax[0].set_xlabel(r"N$_{app}$", fontsize=12, fontweight="bold")
    ax[0].grid(visible=True, which="both", axis="y", linestyle="--", color="black")
    ax[0].set_axisbelow(True)
    ax[0].set_ylim([0, 1])

    ############################################
    ## for figure b
    ############################################
    ## data preprocessing
    # calc carbon
    delay = df_cc["cycles"] * df_cc["tclk"]  # ns
    area = df_cc["area"]  # mm2
    energy = df_cc["energy"]  # pJ
    adp = delay * area
    # calc #PEs
    pe_num = list(df_cc["d1"] * df_cc["d2"] * df_cc["d3"] // 1024)  # unit: k
    mem_size = list(df_cc["mem_size"] // 1024 // 8)  # unit: kB
    voltage = list(df_cc["voltage"])
    x_ticklabels = [f"{pe_num[idx]}k, {mem_size[idx]}kB@{voltage[idx]}V" for idx in range(len(pe_num))]
    ############################################
    ## plot
    lifetime = carbon_parameters["lifetime"]  # unit: ns
    lifetime_candidate = [(i+1) * lifetime for i in range(max_nb_apps)]
    ################
    adp = list(adp)
    energy = list(energy)
    for i in range(len(adp)):
        ax[1].scatter(energy[i], adp[i], s=30, marker="o", edgecolor="black", label=x_ticklabels[i])
    # plot outline for CNADP
    for lifetime_idx in range(len(lifetime_candidate)):
        lifetime_curr = lifetime_candidate[lifetime_idx]
        color_idx = lifetime_idx % len(colors)
        rci = ci_op * lifetime_curr / (ci_em / chip_yield)  # ns * mm2 / pJ
        print(rci)
        adp_min = min(adp)
        adp_max = max(adp)
        energy_min = max(min(energy), adp_min / rci)
        energy_max = max(max(energy), adp_max / rci)

        related_adp_min = energy_min * rci
        related_adp_max = energy_max * rci
        ax[1].plot([0, energy_min], [related_adp_min, 0], linestyle="--", color=colors[color_idx])

    ax[1].set_xlabel("Energy [pJ]", fontsize=12, fontweight="bold")
    ax[1].set_ylabel(u"ADP [mm$^2$$\cdot$ns]", fontsize=12, fontweight="bold")
    ax[1].set_xlim([0, max(energy) * 1.1])
    # ax[1].set_ylim([0, max(adp)*1.1])
    adp_second_max = sorted(set(adp), reverse=True)[1]
    ax[1].set_ylim([0, adp_second_max * 1.1])  # calibration for this experiment
    ax[1].grid(visible=True, which="both", axis="y", linestyle="--", color="black")
    ax[1].set_axisbelow(True)

    # ax[0].legend(ncol=4, loc="upper center")
    plt.tight_layout()
    plt.show()
    pass


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
    # setting for experiment 1
    # pkl_name_list = ["resnet8_1.0.pkl", "resnet8_0.8.pkl", "resnet8_0.6.pkl"]
    # setting for experiment 2
    # pkl_name_list = ["resnet8_1.0.pkl", "resnet8_0.8.pkl", "resnet8_0.6.pkl", "resnet8_0.4.pkl"]
    # setting for experiment 3
    pkl_name_list = ["resnet8_1.0.pkl", "resnet18_1.0.pkl", "resnet50_1.0.pkl",
                     "resnet8_0.8.pkl", "resnet18_0.8.pkl", "resnet50_0.8.pkl",
                     "resnet8_0.6.pkl", "resnet18_0.6.pkl", "resnet50_0.6.pkl"]
    # setting for experiment 4
    # pkl_name_list = ["resnet8_1.0.pkl", "resnet8_0.8.pkl", "resnet8_0.6.pkl", "resnet8_0.4.pkl"]
    ###################################################
    ## carbon setting
    carbon_parameters = {
        "ci_em": 14.13,  # g,CO2/mm2@28nm
        "ci_op": 41 / 3.6e+18,  # g,CO2/pJ
        "chip_yield": 0.95,
        "lifetime": 94608000e+9,  # 3 years in ns
    }
    ###################################################
    ans = []
    for output_pkl_name in pkl_name_list:
        with open(output_folder + output_pkl_name, "rb") as fp:
            ans.append(pickle.load(fp))
    ans_tot = pd.concat(ans, ignore_index=True)
    ## experiment 1: comparison across varied metrics
    # plot_across_varied_foms(df=ans_tot, carbon_parameters=carbon_parameters)
    ## experiment 2: impacts of varied power source or embodied carbon intensity
    # please update to: pkl_name_list = ["resnet8_1.0.pkl", "resnet8_0.8.pkl", "resnet8_0.6.pkl", "resnet8_0.4.pkl"]
    # plot_across_varied_carbon_source(df=ans_tot, carbon_parameters=carbon_parameters)
    ## experiment 3: impacts of Number of #ops
    plot_on_algorithm_impact(df=ans_tot, carbon_parameters=carbon_parameters)
    ## experiment 4: impacts of lifetime
    # plot_lifetime_impact(df=ans_tot, carbon_parameters=carbon_parameters)
