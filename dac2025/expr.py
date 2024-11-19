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
from tqdm import tqdm
from collections import OrderedDict


def save_as_pickle(df, filename):
    with open(filename, "wb") as fp:
        pickle.dump(df, fp)


if __name__ == "__main__":
    """
    Experiment scripts on zigzag (IMC) 
    """
    logging_level = logging.CRITICAL  # logging level
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)
    ###################################################
    ## Parameter setting
    d1_size_candidates = [32, 64, 128]  # Num of columns
    d2_size_candidates = [32, 64, 128]  # Num of rows
    d3_size_candidates = [1, 4, 16]  # Num of macros
    mem_size_candidates = [256 * 1024 * 8, 1 * 1024 * 1024 * 8, 64 * 1024 * 1024 * 8]  # unit: bit
    voltage = 1.0  # unit: V (file updating will weirdly fail if more than 1 element)
    ###################################################
    ## Template setting
    workload_candidates = {
        # "vgg19": "../zigzag/inputs/workload/vgg19.onnx",
        # "dscnn": "../zigzag/inputs/workload/mlperf_tiny/ds_cnn.onnx",
        # "ae": "../zigzag/inputs/workload/mlperf_tiny/deepautoencoder.onnx",
        # "mbv1": "../zigzag/inputs/workload/mlperf_tiny/mobilenet_v1.onnx",
        # "deeplabv3": "../zigzag/inputs/workload/mlperf_mobile/deeplabv3_mnv2_ade20k_inferred.onnx",
        # "ssd": "../zigzag/inputs/workload/mlperf_mobile/ssd_mobilenet_v2_inferred.onnx",
        # "mbbert": "../zigzag/inputs/workload/mlperf_mobile/mobilebert_inferred.onnx",
        # "edgetpu": "../zigzag/inputs/workload/mlperf_mobile/mobilenet_edgetpu_inferred.onnx",
        "resnet50": "../zigzag/inputs/workload/resnet50.onnx",
        "resnet18": "../zigzag/inputs/workload/resnet18.onnx",
        "resnet8": "../zigzag/inputs/workload/mlperf_tiny/resnet8.onnx",
    }
    hardware_file = "../zigzag/inputs/hardware/dimc_cp.yaml"
    mapping_file = "../zigzag/inputs/mapping/default_imc.yaml"
    operand_precision = 8  # unit: bit
    replace_hardware_line_candidates = {"pe_sizes": 108,  # key, line number
                                        "sram_size": 57,
                                        "sram_r_bw": 58,
                                        "sram_w_bw": 59,
                                        "sram_r_cost": 60,
                                        "sram_w_cost": 61,
                                        "sram_area": 62,
                                        "cell_w_cost": 8,
                                        "rf_1b_r_cost": 24,
                                        "rf_1b_w_cost": 25,
                                        "rf_2b_r_cost": 41,
                                        "rf_2b_w_cost": 42,
                                        "dram_r_cost": 83,
                                        "dram_w_cost": 84,
                                        }
    ###################################################
    ## Output setting
    output_folder = "./pkl/"
    os.makedirs(output_folder, exist_ok=True)
    ###################################################
    ## Tech setting
    threshold_voltage = 0.3  # unit: V
    norminal_voltage = 0.9  # unit: V
    nd2_dly_standard = 0.0478  # unit: ns
    xor2_dly_standard = 0.0478 * 2.4  # unit: ns
    dff_area = 0.614 * 6 / 1e6  # unit: mm2
    alpha = 1.3  # alpha in the power law model
    tech_file = "../zigzag/hardware/architecture/imc_unit.py"
    replace_idx_candidates = {"vdd": 16,
                              "nd2_dly": 25,
                              "xor2_dly": 26, }
    ###################################################
    ## Simulation
    TIME_S = time.time()
    for workload_key, workload_filename in workload_candidates.items():
        results = []
        # progress_bar = tqdm(total=len(d1_size_candidates)*len(d2_size_candidates)*len(d3_size_candidates), desc="Simulation", colour="green", ascii=" >=")
        progress_bar = tqdm(total=len(d1_size_candidates), desc="Simulation", colour="green", ascii=" >=")
        for d1, d2, d3 in zip(d1_size_candidates, d2_size_candidates, d3_size_candidates):
            for mem_size in mem_size_candidates:
                assert voltage > threshold_voltage
                TIME_A = time.time()
                # extract cacti memory cost
                bw = max(d2 * d3 * operand_precision, d1 * d3 * 2 * operand_precision)  # unit: bit
                access_time, area, r_cost, w_cost = get_cacti_cost(cacti_path="../zigzag/cacti/cacti_master",
                                                                   tech_node=0.028,
                                                                   mem_type="sram",
                                                                   mem_size_in_byte=mem_size // 8,
                                                                   bw=bw)
                # derive delay scaling results, based on alpha power law model for velocity saturation region
                dly_scaling_factor_standard = norminal_voltage / ((norminal_voltage - threshold_voltage) ** alpha)
                dly_scaling_factor = voltage / ((voltage - threshold_voltage) ** alpha)
                new_nd2_dly = dly_scaling_factor / dly_scaling_factor_standard * nd2_dly_standard
                new_xor2_dly = dly_scaling_factor / dly_scaling_factor_standard * xor2_dly_standard
                # update tech file
                with open(tech_file, "r") as fp:
                    con = fp.readlines()
                for replace_key, replace_idx in replace_idx_candidates.items():
                    text = con[replace_idx]
                    start_idx = text.find(":") + 1
                    end_idx = text.find(",")
                    if replace_key == "vdd":
                        item = voltage
                    elif replace_key == "nd2_dly":
                        item = new_nd2_dly
                    else:
                        item = new_xor2_dly
                    con[replace_idx] = text[:start_idx] + " " + f"{item}" + text[end_idx:]
                with open(tech_file, "w") as fp:
                    for line in con:
                        fp.write(line)
                # update hardware yaml
                with open(hardware_file, "r") as file:
                    con = file.readlines()
                item_collect = {
                    "pe_sizes": [d1, d2, d3],
                    "sram_size": mem_size,
                    "sram_r_bw": bw,
                    "sram_w_bw": bw,
                    "sram_r_cost": r_cost * ((voltage / norminal_voltage)**2),
                    "sram_w_cost": w_cost * ((voltage / norminal_voltage)**2),
                    "sram_area": area,
                    "cell_w_cost": 0.095 * ((voltage / norminal_voltage)**2),
                    "rf_1b_r_cost": 0.021 * ((voltage / norminal_voltage)**2),
                    "rf_1b_w_cost": 0.021 * ((voltage / norminal_voltage)**2),
                    "rf_2b_r_cost": 0.042 * ((voltage / norminal_voltage)**2),
                    "rf_2b_w_cost": 0.042 * ((voltage / norminal_voltage)**2),
                    "dram_r_cost": 240 * ((voltage / norminal_voltage)**2),
                    "dram_w_cost": 240 * ((voltage / norminal_voltage)**2),
                }
                for replace_key, replace_idx in replace_hardware_line_candidates.items():
                    text = con[replace_idx]
                    start_idx = text.find(":") + 1
                    con[replace_idx] = text[:start_idx] + " " + f"{item_collect[replace_key]}" + "\n"
                with open(hardware_file, "w") as file:
                    for line in con:
                        file.write(line)
                # simulation
                energy, latency, tclk, area, cme = get_hardware_performance_zigzag(workload=workload_filename,
                                                                                   accelerator=hardware_file,
                                                                                   mapping=mapping_file,
                                                                                   in_memory_compute=True)
                result = {
                    "workload": workload_key,
                    "d1": d1,
                    "d2": d2,
                    "d3": d3,
                    "mem_size": mem_size,
                    "voltage": voltage,
                    "energy": energy,
                    "cycles": latency,
                    "tclk": tclk,
                    "area": area,
                    "cme": cme,
                }
                results.append(result)
                TIME_B = time.time()
                elapsed_time = round(TIME_B - TIME_A, 1)  # second
                logging.warning(f"workload: {workload_key}, d1/d2/d3: {d1}/{d2}/{d3}, mem: {mem_size},"
                                 f"Vdd: {voltage}, time: {elapsed_time} sec.")
            progress_bar.update(1)
        df = pd.DataFrame(results)
        # save df to pickle
        save_as_pickle(df, output_folder + f"{workload_key}_{voltage}_v2.pkl")
    TIME_E = time.time()
    sim_time = round(TIME_E - TIME_S, 1)
    logging.critical(f"Simulation time: {sim_time} sec.")

    pass
