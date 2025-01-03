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
        self.encoding: dict = {"I": "bm", "W": None, "O": None}
        self.tile_size: dict = {"I": 8, "W": 8}
        self.idx_precision = {
            "I": 0,
            "W": 0,
            "O": 0,
        }

    def derive_idx_precision(self, dense_element_counts: dict, average_density: dict):
        # calc idx precision
        for layer_op in self.idx_precision.keys():
            if layer_op in self.encoding and self.encoding[layer_op] == "bm":
                idx_precision = dense_element_counts[layer_op] / (dense_element_counts[layer_op] * average_density[layer_op])
                self.idx_precision[layer_op] = idx_precision
            else:
                pass

    @staticmethod
    def ceil_distribution_stats(mu, sigma, n_samples=1000000):
        # Generate samples from original Gaussian
        x = np.random.normal(mu, sigma, n_samples)

        # Apply ceiling function
        x_ceil = np.ceil(x)

        # Calculate statistics
        mean_ceil = np.mean(x_ceil)
        std_ceil = np.std(x_ceil)

        return mean_ceil, std_ceil

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
                # if tm_comb[0] in r_loops[layer_op]:
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
                        if loop_dim not in r_loops[layer_op]:
                            loop_combs.append((loop_dim, loop_size))
                        else:
                            allocated_loop_size = np.prod([x[1] for x in loop_combs if x[0] in r_loops[layer_op]])
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

        """ step 7 (Exp): derive the memory utilization of sram_36MB_A (for experiment purpose) """
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
        mem_lats: dict = {  # lat mean
            "I": [],
            "W": [],
            "O": [],
        }
        mem_lats_std: dict = {  # lat std
            "I": [],
            "W": [],
            "O": [],
        }
        mem_ees: dict = {  # ee mean
            "I": [],
            "W": [],
            "O": [],
        }
        mem_ees_std: dict = {  # ee std
            "I": [],
            "W": [],
            "O": [],
        }
        average_density_act: dict = {
            "mean": np.mean(spar_act["density_mean_collect"]),
            "std": np.std(spar_act["density_mean_collect"]),
        }
        for layer_op in ["I", "W", "O"]:
            related_arch_op = memory_operand_links[layer_op]
            r_loop_sm = []
            # calc spatial loops
            for sm_key, sm_value in spatial_unrolling.items():
                related_layer_op = spatial_unrolling_hint[sm_key]
                if related_layer_op in r_loops[layer_op]:
                    r_loop_sm.append((related_layer_op, sm_value))
            sm_loops_size = np.prod([x[1] for x in r_loop_sm])
            for mem_name, mem_info in arch["memories"].items():
                served_arch_ops = mem_info["operands"]
                served_layer_ops = []
                for mapping_layer_op, mapping_arch_op in mapping["memory_operand_links"].items():
                    if mapping_arch_op in served_arch_ops:
                        served_layer_ops.append(mapping_layer_op)
                if layer_op not in served_layer_ops:
                    continue
                # calc spatial served size
                served_arch_dim = mem_info["served_dimensions"]
                served_dim_size = 1
                arch_dim_info = [(arch["operational_array"]["dimensions"][x], arch["operational_array"]["sizes"][x])
                                 for x in range(len(arch["operational_array"]["sizes"]))]
                for arch_dim, arch_dim_size in arch_dim_info:
                    if arch_dim not in served_arch_dim:
                        served_dim_size *= arch_dim_size
                # calc tm loops on lower mem
                tm_loops_size_on_lower_mem = 1
                if mem_name not in [x[0] for x in map_info[layer_op]]:
                    pass
                else:
                    for mem_name_info, tm_loops_info in map_info[layer_op]:
                        if mem_name == mem_name_info:
                            break
                        tm_loops_size_on_lower_mem *= np.prod([x[1] for x in tm_loops_info])
                # calc tm loops on higher mem
                tm_loops_size_on_higher_mem = 1
                curr_mem_index = 0
                recorded_mem_names = [x[0] for x in map_info[layer_op]]
                if mem_name not in recorded_mem_names:
                    pass
                else:
                    curr_mem_index = recorded_mem_names.index(mem_name)
                for index, (mem_name_info, tm_loops_info) in enumerate(map_info[layer_op]):
                    if index >= curr_mem_index:
                        tm_loops_size_on_higher_mem *= np.prod([x[1] for x in tm_loops_info])
                # calc idx
                served_arch_ops = mem_info["operands"]
                if "encoding" in mem_info.keys():
                    encoding_tags = mem_info["encoding"]
                    encoding_tag = encoding_tags[served_arch_ops.index(related_arch_op)]
                else:
                    encoding_tag = "off"
                if encoding_tag != "off":
                    precision_total = layer_op_precision[layer_op] + self.idx_precision[layer_op]
                else:
                    precision_total = layer_op_precision[layer_op]
                mem_bw = mem_info["r_bw"]

                # consider gating impact, calc size bit per transfer
                if layer_op in ["O", "W"] or self.saf[layer_op] == "skipping":
                    size_bit_to_transfer = tm_loops_size_on_lower_mem * sm_loops_size * precision_total
                    tm_loops_size_on_higher_mem = tm_loops_size_on_higher_mem
                else:  # gating
                    size_bit_to_transfer = tm_loops_size_on_lower_mem * sm_loops_size * precision_total * average_density_act["mean"]
                    tm_loops_size_on_higher_mem /= average_density_act["mean"]  # calc corresponding transfer count

                # calc dense bw requirement per transfer
                size_bit_dense = tm_loops_size_on_lower_mem * sm_loops_size * precision_total
                dense_bw = size_bit_dense / served_dim_size

                # calc lat mean and std
                lat_cc_mean = math.ceil(size_bit_to_transfer / (served_dim_size * mem_bw)) * tm_loops_size_on_higher_mem
                lat_cc_std = 0
                if encoding_tag == "off" or layer_op in ["W", "O"]:
                    pass
                else:
                    if self.saf[layer_op] == "gating":
                        # use sampling method to calc std
                        if mem_bw >= dense_bw:
                            lat_cc_std = 0
                        else:
                            vec_cc_mean_before_ceil = size_bit_to_transfer / (served_dim_size * mem_bw)
                            vec_cc_std_before_ceil = size_bit_to_transfer / average_density_act["mean"] * average_density_act[
                                "std"] / (served_dim_size * mem_bw)
                            vec_cc_mean, vec_cc_std =self.ceil_distribution_stats(mu=vec_cc_mean_before_ceil, sigma=vec_cc_std_before_ceil)
                            lat_cc_mean = vec_cc_mean * tm_loops_size_on_higher_mem
                            lat_cc_std = vec_cc_std * tm_loops_size_on_higher_mem
                    else:  # skipping
                        lat_cc_std = lat_cc_mean / average_density_act["mean"] * average_density_act["std"]
                mem_lats[layer_op].append(lat_cc_mean)
                mem_lats_std[layer_op].append(lat_cc_std)

                # calc ee mean and std
                ee_pj = lat_cc_mean * (mem_info["r_cost"] + mem_info["w_cost"])
                ee_pj_std = lat_cc_std * (mem_info["r_cost"] + mem_info["w_cost"])
                mem_ees[layer_op].append(ee_pj)
                mem_ees_std[layer_op].append(ee_pj_std)
                if mem_name == "sram_36MB_A":  # for debugging mode
                    pass

        """ Exp: observe the mem_lats and mem_ees """
        for layer_op in ["I"]:
            mem_lat_max = max(mem_lats[layer_op])
            max_index_pool = []
            for index in range(len(mem_lats[layer_op])):
                if mem_lats[layer_op][index] == mem_lat_max:
                    max_index_pool.append(index)
            mem_lat_std = 0
            max_index = 0
            for index in max_index_pool:
                lat_std = mem_lats_std[layer_op][index]
                if lat_std > mem_lat_std:
                    max_index = index
                    mem_lat_std = lat_std
            mem_lat_std = mem_lats_std[layer_op][max_index]
            mem_ee_mean = mem_ees[layer_op][max_index]
            mem_ee_std = mem_ees_std[layer_op][max_index]
            logging.info(f"[memory bottleneck] lat_mu: {mem_lat_max}, lat_std: {mem_lat_std}, "
                         f"ee_mu: {mem_ee_mean}, ee_std: {mem_ee_std}")

        """ step 9: derive the datapath cost """
        datapath_lats = 0
        datapath_lats_std = 0
        datapath_ees = 0
        datapath_ees_std = 0
        mac_ee_unit = arch["operational_array"]["unit_energy"]
        # calc sparse mac count
        if self.saf["I"] == "gating" and self.saf["W"] == "gating":
            sparse_mac_count = dense_mac_count
            sparse_mac_count_std = 0
        elif self.saf["I"] == "gating" and self.saf["W"] == "skipping":
            sparse_mac_count = dense_mac_count * average_density["W"]
            sparse_mac_count_std = 0
        elif self.saf["I"] == "skipping" and self.saf["W"] == "gating":
            sparse_mac_count = dense_mac_count * average_density["I"]
            sparse_mac_count_std = dense_mac_count * average_density_act["std"]
        elif self.saf["I"] == "skipping" and self.saf["W"] == "skipping":
            sparse_mac_count = dense_mac_count * average_density["I"] * average_density["W"]
            sparse_mac_count_std = dense_mac_count * average_density_act["std"]
        else:  # dense mode
            sparse_mac_count = dense_mac_count
            sparse_mac_count_std = 0
        # get sm loops
        spatial_unrolling_size = np.prod([x for x in spatial_unrolling.values()])
        # calc lat mu
        datapath_lats = sparse_mac_count / spatial_unrolling_size
        # calc lat std
        datapath_lats_std = sparse_mac_count_std / spatial_unrolling_size
        # calc ee mu
        sparse_mac_count_ee = dense_mac_count
        sparse_mac_count_ee_std = 0
        if self.saf["I"] in ["gating", "skipping"]:
            sparse_mac_count_ee *= average_density["I"]
            sparse_mac_count_ee_std = average_density_act["std"] * dense_mac_count
        if self.saf["W"] in ["gating", "skipping"]:
            sparse_mac_count_ee *= average_density["W"]
            sparse_mac_count_ee_std *= average_density["W"]
        datapath_ees = sparse_mac_count_ee * mac_ee_unit
        # calc ee std
        datapath_ees_std = sparse_mac_count_ee_std * mac_ee_unit
        logging.info(f"[datapath] lat_mu: {datapath_lats}, lat_std: {datapath_lats_std}, "
                     f"ee_mu: {datapath_ees}, ee_std: {datapath_ees_std}")

        """ step 10: derive the total cost """


        """ step 11: prepare the output """
        pass


if __name__ == "__main__":
    logging_level = logging.INFO  # logging level
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)
    exp = exp_sigma()
    exp.simulation()
