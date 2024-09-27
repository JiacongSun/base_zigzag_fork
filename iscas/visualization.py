import os
import sys

sys.path.insert(0, os.path.dirname(os.getcwd()))

from zigzag.hardware.architecture.get_cacti_cost import get_cacti_cost
import matplotlib.pyplot as plt
import numpy as np
import logging


def plot_mem_comparison():
    """
    plot mem comparison for Bowen
    """
    ###############################
    # setting
    bar_width = 0.2
    ###############################
    # initialization
    info = {mem: {} for mem in ["dram", "sram", "bowen"]}
    mem_size_list = [32 * 1024, 1024 * 1024, 32 * 1024 * 1024, 1024 * 1024 * 1024]
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
            if mem_size < 1024*1024:
                labels.append(f"{mem_size//1024}K")
            elif mem_size < 1024*1024*1024:
                labels.append(f"{mem_size // 1024 // 1024}M")
            else:
                labels.append(f"{mem_size//1024//1024//1024}G")
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


if __name__ == "__main__":
    # for Bowen's ISCAS paper
    logging_level = logging.WARN  # logging level
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)
    #########################################
    # Experiment 1: area, energy comparison among SRAM and Bowen's mem
    # comment to bowen:
    # (1) is it correct that E/bit does not scale with memory size for our case?
    # (2) is the area trend and value make sense?
    plot_mem_comparison()
    #########################################
    # Experiment 2: zigzag evaluation result
    pass
