from exp_mem_util import derive_idx_precision
import math
import numpy as np
from sigma_like import exp_sigma
import logging
import matplotlib.pyplot as plt


def plot_mem(bw, lat_mu_gating, lat_std_gating, ee_mu_gating, ee_std_gating,
             lat_mu_skipping, lat_std_skipping, ee_mu_skipping, ee_std_skipping):
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    gating_color = '#4B88B5'  # Soft blue
    skipping_color = '#D17575'  # Soft red

    # Plot latency data
    ax1.errorbar(bw, lat_mu_gating, yerr=lat_std_gating, fmt='o-', label='Gating', capsize=5, color=gating_color)
    ax1.errorbar(bw, lat_mu_skipping, yerr=lat_std_skipping, fmt='s-', label='Skipping',
                 capsize=5, color=skipping_color)
    ax1.set_xlabel('Bandwidth', fontsize=12, weight='bold')
    ax1.set_ylabel('Latency (cc)', fontsize=12, weight='bold')
    ax1.set_title('Latency vs Bandwidth', fontsize=12, weight='bold')
    ax1.grid(True)
    ax1.set_axisbelow(True)
    ax1.legend()

    # Plot energy efficiency data
    ax2.errorbar(bw, ee_mu_gating, yerr=ee_std_gating, fmt='o-', label='Gating', capsize=5, color=gating_color)
    ax2.errorbar(bw, ee_mu_skipping, yerr=ee_std_skipping, fmt='s-', label='Skipping', capsize=5, color=skipping_color)
    ax2.set_xlabel('Bandwidth', fontsize=12, weight='bold')
    ax2.set_ylabel('Energy (pJ)', fontsize=12, weight='bold')
    ax2.set_title('Energy vs Bandwidth', fontsize=12, weight='bold')
    ax2.grid(True)
    ax2.set_axisbelow(True)
    ax2.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plot
    plt.show()


if __name__ == "__main__":
    """
    Exp: mem utilization (512 KB) @ L2, ResNet18
    """
    logging_level = logging.INFO  # logging level
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)
    mem_size_512kb = 512 * 1024 * 8
    ox = 56
    oy = 56
    c = 64
    fx = 3
    fy = 3
    k = 64
    op_pres = 8  # INT8
    encoding = "bm"  # bitmasking
    tile_size = 8  # storage tile size
    average_density = 0.59813
    density_std = 0.08958
    weight_density = 0.923  # on C dim
    # for lat/ee calc
    saf_pool = ["gating", "skipping"]
    mem_bw = 8  # bit
    sm_unrolling = 8  # parfor C: 8
    ir_sm_unrolling = 8  # parfor K: 8
    ir_tm_unrolling = k * fx * fy / ir_sm_unrolling
    r_costs = {46: 19.17, 47: 19.31, 54: 25.3480134375, 61: 26.422460718750003, 64: 26.77,
               68: 27.310213125, 78: 28.841867437500007, 89: 30.447887343750004}  # pj
    w_costs = {46: 15, 47: 15.8, 54: 16.3419474375, 61: 17.34167475, 64: 17.71,
               68: 18.190342125, 78: 19.61375715, 89: 21.13764406875}  # pj

    encoding_pool = ["bm"]
    tile_size_pool = [128]
    bw_pool = [47, 54, 61, 68, 78]

    dense_element_counts = ox * oy * c * weight_density
    idx_precision = derive_idx_precision(encoding, tile_size, dense_element_counts, average_density)
    bw_aver_density = math.ceil(sm_unrolling * (op_pres + idx_precision) * average_density)
    bw_1sigma_density = math.ceil(sm_unrolling * (op_pres + idx_precision) * (average_density + 1 * density_std))
    bw_2sigma_density = math.ceil(sm_unrolling * (op_pres + idx_precision) * (average_density + 2 * density_std))
    bw_3sigma_density = math.ceil(sm_unrolling * (op_pres + idx_precision) * (average_density + 3 * density_std))
    bw_6sigma_density = math.ceil(sm_unrolling * (op_pres + idx_precision) * (average_density + 6 * density_std))
    bw_dense = math.ceil(sm_unrolling * (op_pres + idx_precision))
    logging.info(f"bw_aver_density: {bw_aver_density}, bw_1sigma: {bw_1sigma_density}, bw_2sigma: {bw_2sigma_density},"
                 f"bw_3sigma: {bw_3sigma_density}, bw_6sigma: {bw_6sigma_density}, bw_dense: {bw_dense}")

    lat_mu_gating = []
    lat_std_gating = []
    ee_mu_gating = []
    ee_std_gating = []
    lat_mu_skipping = []
    lat_std_skipping = []
    ee_mu_skipping = []
    ee_std_skipping = []
    for saf in saf_pool:
        assert saf in ["gating", "skipping"]
        for mem_bw in bw_pool:
            for tile_size in tile_size_pool:
                dense_element_bit = dense_element_counts * op_pres
                size_occupied_bit = dense_element_counts * average_density * (op_pres + idx_precision)
                # calc lat_cc mean and std
                if saf == "gating":
                    size_bit_each_transfer = sm_unrolling * (op_pres + idx_precision) * average_density
                    tm_loops_size_on_higher_mem = dense_element_counts / sm_unrolling * ir_tm_unrolling
                    transfer_cc_mu_before_ceiling = size_bit_each_transfer / mem_bw
                    transfer_cc_std_before_ceiling = sm_unrolling * (op_pres + idx_precision) * density_std / mem_bw
                    transfer_cc_mu, transfer_cc_std = exp_sigma.ceil_distribution_stats(
                        mu=transfer_cc_mu_before_ceiling,
                        sigma=transfer_cc_std_before_ceiling)
                    lat_cc_int = transfer_cc_mu * tm_loops_size_on_higher_mem
                    lat_cc_std = transfer_cc_std * tm_loops_size_on_higher_mem
                    if mem_bw >= bw_dense:
                        lat_cc_std = 0
                else:  # skipping
                    size_bit_each_transfer = sm_unrolling * (op_pres + idx_precision)
                    tm_loops_size_on_higher_mem = dense_element_counts * average_density / sm_unrolling
                    tm_loops_size_on_higher_mem_std = dense_element_counts * density_std / sm_unrolling
                    lat_cc_int = math.ceil(size_bit_each_transfer / mem_bw) * tm_loops_size_on_higher_mem
                    lat_cc_std = math.ceil(size_bit_each_transfer / mem_bw) * tm_loops_size_on_higher_mem_std
                # calc ee mean and std
                ee_mean = lat_cc_int * (r_costs[mem_bw] + w_costs[mem_bw])
                ee_std = lat_cc_std * (r_costs[mem_bw] + w_costs[mem_bw])
                logging.info(f"sav: {saf}, bw: {mem_bw}, lat_cc: {lat_cc_int}, lat_std: {lat_cc_std}, ee_cc: {ee_mean}, ee_std: {ee_std}")
                if saf == "gating":
                    lat_mu_gating.append(lat_cc_int)
                    lat_std_gating.append(lat_cc_std)
                    ee_mu_gating.append(ee_mean)
                    ee_std_gating.append(ee_std)
                else:
                    lat_mu_skipping.append(lat_cc_int)
                    lat_std_skipping.append(lat_cc_std)
                    ee_mu_skipping.append(ee_mean)
                    ee_std_skipping.append(ee_std)
    plot_mem(bw_pool, lat_mu_gating, lat_std_gating, ee_mu_gating, ee_std_gating,
             lat_mu_skipping, lat_std_skipping, ee_mu_skipping, ee_std_skipping)