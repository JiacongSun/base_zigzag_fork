import os
import sys

sys.path.insert(0, os.path.dirname(os.getcwd()))

from zigzag.hardware.architecture.get_cacti_cost import get_cacti_cost
import matplotlib.pyplot as plt
import numpy as np
import logging
import yaml
import pickle
from zigzag.api import get_hardware_performance_zigzag


def plot_mem_comparison(mem_list: list = ["dram", "sram", "bowen"],
                        mem_size_list: list = [32 * 1024, 1024 * 1024, 32 * 1024 * 1024, 1024 * 1024 * 1024],
                        bar_width: float = 0.2):
    """
    plot mem comparison for Bowen
    :param mem_list: targeted mem types
    :param mem_size_list: targeted mem size in byte
    :param bar_width: bar width in the plots
    :return info (dict): area, energy_wr, energy_rd, access time information
    """
    ###############################
    # initialization
    info = {mem: {} for mem in mem_list}
    for bw in [128]:
        for mem_size in mem_size_list:  # unit: B
            # initialize key with mem_size
            info["dram"][f"{mem_size}"] = {}
            info["sram"][f"{mem_size}"] = {}
            info["bowen"][f"{mem_size}"] = {}
            # calc dram result (normalized from dramsim, config: DDR3_1Gb_x8_1333.ini
            area = 64 / (1024 * 1024 * 1024) * mem_size
            access_time = 0  # NA
            r_cost_per_bit = 3.7  # pJ/bit
            w_cost_per_bit = 3.7  # pJ/bit
            info["dram"][f"{mem_size}"]["time"] = access_time
            info["dram"][f"{mem_size}"]["area"] = area
            info["dram"][f"{mem_size}"]["r_cost_per_bit"] = r_cost_per_bit
            info["dram"][f"{mem_size}"]["w_cost_per_bit"] = w_cost_per_bit
            # extract sram result
            access_time, area, r_cost, w_cost = get_cacti_cost(cacti_path="../zigzag/cacti/cacti_master",
                                                               tech_node=0.028,
                                                               mem_type="sram",
                                                               mem_size_in_byte=mem_size,
                                                               bw=bw)
            r_cost_per_bit = r_cost / bw
            w_cost_per_bit = w_cost / bw
            info["sram"][f"{mem_size}"]["time"] = access_time
            info["sram"][f"{mem_size}"]["area"] = area
            info["sram"][f"{mem_size}"]["r_cost_per_bit"] = r_cost_per_bit
            info["sram"][f"{mem_size}"]["w_cost_per_bit"] = w_cost_per_bit
            # extract Bowen's result (16 stacks)
            area_per_bit = 4 * 100 * 100  # nm2
            num_stack = 16
            area = mem_size * 8 / num_stack * area_per_bit / 1e+12  # mm2
            access_time = 4  # ns
            r_cost_per_bit = (138 // 2) / 1000  # pJ/bit

            info["bowen"][f"{mem_size}"]["time"] = access_time
            info["bowen"][f"{mem_size}"]["area"] = area
            info["bowen"][f"{mem_size}"]["r_cost_per_bit"] = r_cost_per_bit
            info["bowen"][f"{mem_size}"]["w_cost_per_bit"] = r_cost_per_bit
    ###############################
    # plot: area, wr_energy, rd_energy
    plot_num = 3
    fig, axs = plt.subplots(figsize=(8, 4), nrows=1, ncols=plot_num)
    index = np.arange(len(mem_size_list))
    # fig 1: area
    area_dram = [info["dram"][f"{mem_size}"]["area"] for mem_size in mem_size_list]
    area_sram = [info["sram"][f"{mem_size}"]["area"] for mem_size in mem_size_list]
    area_bowen = [info["bowen"][f"{mem_size}"]["area"] for mem_size in mem_size_list]
    axs[0].bar(index - bar_width, area_dram, edgecolor="black", width=bar_width, facecolor="orange", hatch="//",
               label="DDR3")
    axs[0].bar(index, area_sram, edgecolor="black", width=bar_width, facecolor="green", hatch="--", label="SRAM")
    axs[0].bar(index + bar_width, area_bowen, edgecolor="black", width=bar_width, facecolor="yellow", hatch="\\\\",
               label="Ours")
    # fig 2: wr_energy
    wr_dram = [info["dram"][f"{mem_size}"]["w_cost_per_bit"] for mem_size in mem_size_list]
    wr_sram = [info["sram"][f"{mem_size}"]["w_cost_per_bit"] for mem_size in mem_size_list]
    wr_bowen = [info["bowen"][f"{mem_size}"]["w_cost_per_bit"] for mem_size in mem_size_list]
    axs[1].bar(index - bar_width, wr_dram, edgecolor="black", width=bar_width, facecolor="orange", hatch="//",
               label="DDR3")
    axs[1].bar(index, wr_sram, edgecolor="black", width=bar_width, facecolor="green", hatch="--", label="SRAM")
    axs[1].bar(index + bar_width, wr_bowen, edgecolor="black", width=bar_width, facecolor="yellow", hatch="\\\\",
               label="Ours")
    # fig 3: rd_energy
    rd_dram = [info["dram"][f"{mem_size}"]["r_cost_per_bit"] for mem_size in mem_size_list]
    rd_sram = [info["sram"][f"{mem_size}"]["r_cost_per_bit"] for mem_size in mem_size_list]
    rd_bowen = [info["bowen"][f"{mem_size}"]["r_cost_per_bit"] for mem_size in mem_size_list]
    axs[2].bar(index - bar_width, rd_dram, edgecolor="black", width=bar_width, facecolor="orange", hatch="//",
               label="DDR3")
    axs[2].bar(index, rd_sram, edgecolor="black", width=bar_width, facecolor="green", hatch="--", label="SRAM")
    axs[2].bar(index + bar_width, rd_bowen, edgecolor="black", width=bar_width, facecolor="yellow", hatch="\\\\",
               label="Ours")
    ###############################
    # configuration
    # change x tick label
    for x in range(plot_num):
        axs[x].set_xticks(index)
        labels = []
        for mem_size in mem_size_list:
            if mem_size < 1024 * 1024:
                labels.append(f"{mem_size // 1024}K")
            elif mem_size < 1024 * 1024 * 1024:
                labels.append(f"{mem_size // 1024 // 1024}M")
            else:
                labels.append(f"{mem_size // 1024 // 1024 // 1024}G")
        axs[x].set_xticklabels(labels)
    # add legend and grid
    for x in range(plot_num):
        if x == 0:
            axs[x].legend(fontsize=12)
        axs[x].grid(which="major", axis="y", color="gray", linestyle="--", linewidth=1)
        axs[x].set_axisbelow(True)
    # add label
    for x in range(plot_num):
        axs[x].set_xlabel("Memory size [B]", fontsize=14)
    axs[0].set_ylabel("Area [mm$^2$]", fontsize=14)
    axs[1].set_ylabel("Write energy [pJ/bit]", fontsize=14)
    axs[2].set_ylabel("Read energy [pJ/bit]", fontsize=14)
    # add title
    fig.suptitle(f"bw: {bw}-bit", fontsize=14)
    # switch y to logy
    for x in range(plot_num):
        axs[x].set_yscale("log")
    plt.tight_layout()
    # plt.show()
    # save plt
    # os.makedirs("./iscas/", exist_ok=True)
    plt.savefig("./mem_comparison.png", dpi=300, bbox_inches="tight")
    return info


def zigzag_evaluation():
    """
    evaluate cost on zigzag
    """
    ###############################
    # setting
    mem_list = ["dram", "sram", "bowen"]
    ###############################
    # zigzag setting
    workloads = {
        "alexnet": "zigzag/inputs/workload/alexnet.onnx",
        "resnet18": "zigzag/inputs/workload/resnet18.onnx",
        "mobilenetv2": "zigzag/inputs/workload/mobilenetv2.onnx",
        "resnet50": "zigzag/inputs/workload/resnet50.onnx",
        "vgg19": "zigzag/inputs/workload/vgg19.onnx"}
    mapping = "zigzag/inputs/mapping/default_imc.yaml"
    accelerator = "zigzag/inputs/hardware/dimc.yaml"
    # required top-level weight memory size
    required_weight_in_byte = {"alexnet": 60954656,
                               "resnet18": 11678912,
                               "mobilenetv2": 3469760,
                               "resnet50": 23454912,
                               "vgg19": 143652544}
    # current working directory
    cwd = os.getcwd()
    ###############################
    required_weight_in_byte_rounded = {"alexnet": 64 * 1024 * 1024,
                                       "resnet18": 16 * 1024 * 1024,
                                       "mobilenetv2": 4 * 1024 * 1024,
                                       "resnet50": 32 * 1024 * 1024,
                                       "vgg19": 256 * 1024 * 1024}
    # required mem size (byte) per workload
    mem_size_list = list(required_weight_in_byte_rounded.values())
    # catch mem performance
    mem_info = plot_mem_comparison(mem_list=mem_list,
                                   mem_size_list=mem_size_list)
    ###############################
    # change working directory
    os.chdir("../")
    results = {mem: {} for mem in mem_list}
    for workload_name, workload_onnx in workloads.items():
        for mem in mem_list:
            results[mem][workload_name] = {}
            # update input dimc
            yaml_path = "./zigzag/inputs/hardware/dimc.yaml"
            with open(yaml_path, "r") as file:
                data = yaml.safe_load(file)
            # update energy
            required_mem_size = f"{required_weight_in_byte_rounded[workload_name]}"
            r_cost = mem_info[mem][required_mem_size]["r_cost_per_bit"] * 128  # fixed bw
            w_cost = mem_info[mem][required_mem_size]["w_cost_per_bit"] * 128  # fixed bw
            data["memories"]["dram"]["r_cost"] = r_cost
            data["memories"]["dram"]["w_cost"] = w_cost
            area_weight_mem = mem_info[mem][required_mem_size]["area"]
            with open(yaml_path, "w") as file:
                yaml.dump(data, file, sort_keys=False)
            # run zigzag
            energy, latency, tclk, area, cme = get_hardware_performance_zigzag(workload_onnx, accelerator, mapping)
            # calc total cost
            area_total = area + area_weight_mem
            results[mem][workload_name]["area"] = area_total
            results[mem][workload_name]["energy"] = energy
            results[mem][workload_name]["tclk"] = tclk
            results[mem][workload_name]["latency"] = latency
            results[mem][workload_name]["cme"] = cme
    # change-back working directory
    os.chdir(cwd)
    # save results to pkl
    pkl_filename = "./results.pkl"
    with open(pkl_filename, "wb") as file:
        pickle.dump(results, file)


if __name__ == "__main__":
    # for Bowen's ISCAS paper
    logging_level = logging.INFO  # logging level
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)
    #########################################
    # Experiment 1: area, energy comparison among SRAM and Bowen's mem
    # comment to bowen:
    # (1) is it correct that E/bit does not scale with memory size for our case?
    # (2) is the area trend and value make sense?
    # plot_mem_comparison()
    #########################################
    # Experiment 2: zigzag evaluation result
    zigzag_evaluation()
    pass
