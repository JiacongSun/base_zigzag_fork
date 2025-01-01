import pickle
import time
import sys
import os
import logging
import yaml
import numpy as np
import copy
import math

# append the parent folder to the environment
sys.path.insert(0, os.path.dirname(os.getcwd()))

"""
Exp1: the impact of tile size on sigma-like (CG-A, SK-W) arch
"""


class exp_sigma:
    def __init__(self):
        """ Global settings """
        self.model_name = "resnet18"
        self.dataset_name = "imagenet"
        self.layer_id = 2
        self.saf: dict = {"I": "gating", "W": "skipping"}
        self.tm_ordering: tuple = ("OX", "OY", "C", "FX", "FY", "K")  # bottom-to-top
        self.encoding: dict = {"I": "bm", "W": None}
        self.tile_size: dict = {"I": 8, "W": 8}
        self.idx_precision = {
            "I": 0,
            "W": 0,
            "O": 0,
        }

    def derive_idx_precision(self, dense_element_counts: dict, average_density: dict):
        # calc idx precision
        for layer_op in self.idx_precision.keys():
            if layer_op in self.encoding and self.encoding[layer_op] == "rle":
                idx_precision_within_tile = math.log2(self.tile_size[layer_op])
                idx_precision_across_tile = math.log2(
                    dense_element_counts[layer_op] / self.tile_size[layer_op])  # DBB format
                self.idx_precision[layer_op] = idx_precision_within_tile + idx_precision_across_tile
            elif layer_op in self.encoding and self.encoding[layer_op] == "rle":
                self.idx_precision[layer_op] = dense_element_counts[layer_op] / (
                            dense_element_counts[layer_op] * average_density[layer_op])
            else:
                pass

    def simulation(self):
        """ step 1: load in the sparsity data """
        pkl_act = f"../../zigzag/density_parser/pkl/act/{self.dataset_name}/{self.model_name}/" \
                  f"dist_{self.model_name}_{self.dataset_name}_layer{self.layer_id}_tile{self.tile_size['I']}.pkl"
        pkl_weight = f"../../zigzag/density_parser/pkl/weight/{self.model_name}/" \
                     f"dist_{self.model_name}_layer{self.layer_id}_tile{self.tile_size['W']}.pkl"
        with open(pkl_act, "rb") as fp:
            con: list = pickle.load(fp)
            spar_act: dict = {
                "density_list_collect": con[0],
                "density_occurrence_collect": con[1],
                "aver_density_dist": con[2],
                "density_mean_collect": con[3],
                "density_std_collect": con[4],
                "density_covariance_matrix": con[5],
            }
        with open(pkl_weight, "rb") as fp:
            con: list = pickle.load(fp)
            spar_weight: dict = {
                "density_list": con[0],
                "density_occurrence": con[1],
                "density_mean": con[2],
                "density_std": con[3],
            }
        average_density: dict = {
            "I": np.mean(spar_act["density_mean_collect"]),
            "W": spar_weight["density_mean"]
        }

        """ step 2: load in the hw arch and mapping """
        with open("hardware/sigma.yaml", "r") as fp:
            arch: dict = yaml.safe_load(fp)
        with open("mapping/sigma.yaml", "r") as fp:
            mapping: dict = yaml.safe_load(fp)[0]
        memory_operand_links: dict = mapping["memory_operand_links"]  # layer_op: arch_op
        layer_operand_links: dict = {}  # arch_op: layer_op
        for layer_op, arch_op in memory_operand_links.items():
            layer_operand_links[arch_op] = layer_op

        """ step 3: load in the network shape """
        # layer shape for resnet18, layer2
        workload: dict = {"K": 64,
                          "C": 64,
                          "OX": 56,
                          "OY": 56,
                          "FX": 3,
                          "FY": 3}
        r_loops: dict = {"I": ("C", "OX", "OY"),
                         "W": ("C", "K", "FX", "FY"),
                         "O": ("K", "OX", "OY")}
        dense_mac_count = np.prod([size for dim, size in workload.items()])
        dense_element_counts: dict = {
            "I": np.prod([size for dim, size in workload.items() if dim in r_loops["I"]]),
            "W": np.prod([size for dim, size in workload.items() if dim in r_loops["W"]]),
            "O": np.prod([size for dim, size in workload.items() if dim in r_loops["O"]]),
        }
        self.derive_idx_precision(dense_element_counts, average_density)
        for layer_op in ["I", "W"]:
            if layer_op in self.saf.keys() and self.saf[layer_op] is not None:
                # scale layer shape
                for dim_to_scale in r_loops[layer_op]:
                    if workload[dim_to_scale] * average_density[layer_op] < 1:
                        continue
                    else:
                        ori_dim_size = workload[dim_to_scale]
                        workload[dim_to_scale] *= average_density[layer_op]
                        logging.info(
                            f"scaling loop {dim_to_scale} from {ori_dim_size} to {workload[dim_to_scale]} due to average density {average_density[layer_op]} ({layer_op})")
                        break
        sparse_mac_count = np.prod([size for dim, size in workload.items()])

        """ step 4: set the spatial mapping hint """
        spatial_unrolling_hint: dict = {
            "D1": "K",
            "D2": "C"
        }

        """ step 5: calc the spatial mapping """
        sm_dim_d1 = spatial_unrolling_hint["D1"]
        sm_dim_d2 = spatial_unrolling_hint["D2"]
        arch_size_d1 = arch["operational_array"]["sizes"][0]
        arch_size_d2 = arch["operational_array"]["sizes"][1]
        pe_count = arch_size_d1 * arch_size_d2
        layer_dim_d1 = workload[sm_dim_d1]
        layer_dim_d2 = workload[sm_dim_d2]
        spatial_unrolling: dict = {
            "D1": 0,
            "D2": 0,
        }
        if self.saf["W"] == "skipping":
            # principle: first unroll a layer loop if the arch size allows, to maximize the data reuse
            if layer_dim_d2 <= arch_size_d2:
                spatial_unrolling["D2"] = min(arch_size_d2, layer_dim_d2)
                spatial_unrolling["D1"] = min(arch_size_d1 * arch_size_d2 / spatial_unrolling["D2"], layer_dim_d1)
            else:
                spatial_unrolling["D1"] = min(arch_size_d1, layer_dim_d1)
                spatial_unrolling["D2"] = min(arch_size_d1 * arch_size_d2 / spatial_unrolling["D1"], layer_dim_d2)
        else:
            spatial_unrolling["D1"] = min(arch_size_d1, layer_dim_d1)
            spatial_unrolling["D2"] = min(arch_size_d2, layer_dim_d2)

        """ step 6: calc the temporal mapping """
        # dataflow generator
        tm_loops: dict = copy.deepcopy(workload)
        for key, value in spatial_unrolling.items():
            layer_dim_sm = spatial_unrolling_hint[key]
            tm_loops[layer_dim_sm] = tm_loops[layer_dim_sm] / value
        dataflow_gen = []
        for layer_dim in self.tm_ordering:
            dataflow_gen.append((layer_dim, tm_loops[layer_dim]))
        # dataflow mapper
        layer_op_precision: dict = {
            "I": arch["operational_array"]["input_precision"][0],
            "W": arch["operational_array"]["input_precision"][1],
            "O": arch["operational_array"]["input_precision"][0] + arch["operational_array"]["input_precision"][1],
        }
        map_info: dict = {  # record layer_op: (mem_level, tm_loops)
            "I": [],
            "W": [],
            "O": []
        }
        for layer_op in ["I", "W", "O"]:
            related_arch_op = memory_operand_links[layer_op]
            r_loop_to_allocate = []
            r_loop_sm = []
            r_loop_done = []
            for tm_comb in dataflow_gen:
                if tm_comb[0] in r_loops[layer_op]:
                    r_loop_to_allocate.append(tm_comb)
            for sm_key, sm_value in spatial_unrolling.items():
                related_layer_op = spatial_unrolling_hint[sm_key]
                if related_layer_op in r_loops[layer_op]:
                    r_loop_sm.append((related_layer_op, sm_value))
            for mem_name, mem_info in arch["memories"].items():
                served_arch_ops = mem_info["operands"]
                served_layer_ops = []
                for mapping_layer_op, mapping_arch_op in mapping["memory_operand_links"].items():
                    if mapping_arch_op in served_arch_ops:
                        served_layer_ops.append(mapping_layer_op)
                if layer_op not in served_layer_ops:
                    continue
                # calc stored element count
                if "encoding" in mem_info.keys():
                    encoding_tags = mem_info["encoding"]
                    encoding_tag = encoding_tags[served_arch_ops.index(related_arch_op)]
                else:
                    encoding_tag = "off"
                if encoding_tag != "off":
                    mem_size = mem_info["size"] / (layer_op_precision[layer_op] + self.idx_precision[layer_op])
                else:
                    mem_size = mem_info["size"] / layer_op_precision[layer_op]
                assert mem_size >= 1, f"{mem_name} is too small."
                if mem_size == 1:
                    continue
                else:
                    served_arch_dim = mem_info["served_dimensions"]
                    served_dim_size = 1
                    arch_dim_info = [(arch["operational_array"]["dimensions"][x], arch["operational_array"]["sizes"][x])
                                     for x in range(len(arch["operational_array"]["sizes"]))]
                    for arch_dim, arch_dim_size in arch_dim_info:
                        if arch_dim not in served_arch_dim:
                            served_dim_size *= arch_dim_size
                    allowed_loop_size = mem_size * served_dim_size / np.prod([x[1] for x in r_loop_sm]) / np.prod(
                        [x[1] for x in r_loop_done])
                    loop_combs = []
                    r_loop_to_allocate_new = []
                    for idx in range(len(r_loop_to_allocate)):
                        loop_dim = r_loop_to_allocate[idx][0]
                        loop_size = r_loop_to_allocate[idx][1]
                        allocated_loop_size = np.prod([x[1] for x in loop_combs])
                        if allowed_loop_size / (loop_size * allocated_loop_size) >= 1:
                            loop_combs.append((loop_dim, loop_size))
                        else:
                            loop_combs.append((loop_dim, allowed_loop_size / allocated_loop_size))
                            r_loop_to_allocate_new.append(
                                (loop_dim, loop_size - allowed_loop_size / allocated_loop_size))
                            r_loop_to_allocate_new += r_loop_to_allocate[idx + 1:]
                    # reformat r_loop_to_allocate
                    r_loop_to_allocate = r_loop_to_allocate_new
                    r_loop_done = r_loop_done + loop_combs
                    map_info[layer_op].append((mem_name, loop_combs))

        """ step 7: derive the memory utilization """
        for layer_op in ["I"]:
            targeted_mem_info: dict = arch["memories"]["sram_36MB_A"]
            targeted_tm: list = [x[1] for x in map_info[layer_op] if x[0] == "sram_36MB_A"][0]
            targeted_sm: list = []
            for sm_key, sm_value in spatial_unrolling.items():
                related_layer_op = spatial_unrolling_hint[sm_key]
                if related_layer_op in r_loops[layer_op]:
                    targeted_sm.append((related_layer_op, sm_value))
            size_occupied_element = np.prod([size for dim, size in targeted_tm]) * np.prod(
                [size for dim, size in targeted_sm])
            idx_precision = self.idx_precision[layer_op]
            op_precision = arch["operational_array"]["input_precision"][0]
            size_occupied_bit = size_occupied_element * (idx_precision + op_precision)
            average_density_act: dict = {
                "mean": np.mean(spar_act["density_mean_collect"]),
                "std": np.std(spar_act["density_mean_collect"]),
            }
            mem_size_bit = targeted_mem_info["size"]
            curr_util: dict = {
                "mean": size_occupied_bit / mem_size_bit,
                "std": size_occupied_bit / average_density_act["mean"] * average_density_act["std"] / mem_size_bit,
            }
            logging.info(f"Sz_tile(act): {self.tile_size[layer_op]}, util_mean: {curr_util['mean']}, util_std: {curr_util['std']}")
        pass

        """ step 8: derive the memory cost """
        mem_lats: dict = {
            "I": [],
            "W": [],
            "O": [],
        }
        mem_ees: dict = {
            "I": [],
            "W": [],
            "O": [],
        }

        """ step 9: derive the datapath cost """

        """ step 10: derive the total cost """

        """ step 11: prepare the output """
        pass


if __name__ == "__main__":
    logging_level = logging.INFO  # logging level
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)
    exp = exp_sigma()
    exp.simulation()
