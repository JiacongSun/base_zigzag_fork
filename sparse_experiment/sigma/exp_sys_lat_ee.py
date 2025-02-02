import copy

from exp_mem_util import derive_idx_precision
import math
import numpy as np
from sigma_like import exp_sigma
import logging
import matplotlib.pyplot as plt
import pickle
from zigzag.hardware.architecture.get_cacti_cost import get_cacti_cost


def plot_multiple_gaussians(means, stds, labels=None, xlim: float or None = None):
    # Create a suitable x range that covers all distributions
    x = np.linspace(min(means) - 4 * max(stds), max(means) + 4 * max(stds), 1000)

    # Plot each Gaussian distribution
    plt.figure(figsize=(6, 4))
    if labels is None:
        labels_fig = [None] * len(means)
    else:
        labels_fig = labels
    for mu, sigma, label in zip(means, stds, labels_fig):
        # Calculate the Gaussian distribution
        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        plt.fill_between(x, 0, y, alpha=0.3)
        plt.plot(x, y, label=label)

    # plt.title('Multiple Gaussian Distributions')
    plt.xlabel('Latency [ms]', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    if xlim is not None:
        plt.xlim([0, xlim])
    else:
        plt.xlim(left=0)
    # plt.ylim(bottom=-0.1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()


def get_memory_cost(mem_size_in_byte: int, mem_bw_in_bit: int, return_area: bool = False):
    """
    derive the memory w_cost and r_cost from cacti
    :param mem_size_in_byte: memory size in byte
    :param mem_bw_in_bit: memory bandwidth in bit
    :param return_area: if return area
    """
    cacti_path = "../../zigzag/cacti/cacti_master"
    tech_node = 0.028
    access_time, area, r_cost, w_cost = get_cacti_cost(
        cacti_path=cacti_path,
        tech_node=tech_node,
        mem_type="sram",
        mem_size_in_byte=mem_size_in_byte,
        bw=mem_bw_in_bit,
    )
    if return_area:
        return r_cost, w_cost, area
    else:
        return r_cost, w_cost


def derive_6_sigma_bw(ox, oy, c, weight_density, average_density, density_std, sm_unrolling,
                      encoding: str = "bm", tile_size: int = 8, op_pres: int = 8):
    """
    derive the average bw, 1-sigma to 6-sigma bw, dense bw
    :param ox, oy, c: act layer dimension size
    :param weight_density: average weight density
    :param average_density: average act density
    :param density_std: act density std
    :param sm_unrolling: act r-loop spatial mapping size
    :param encoding: compression encoding
    :param tile_size: tile size for compression
    :param op_pres: op precision (8 for INT8)
    """
    dense_element_counts = ox * oy * c * weight_density
    idx_precision = derive_idx_precision(encoding, tile_size, dense_element_counts, average_density)
    bw_aver_density = math.ceil(sm_unrolling * (op_pres + idx_precision) * average_density)
    bw_1sigma_density = math.ceil(sm_unrolling * (op_pres + idx_precision) * (average_density + 1 * density_std))
    bw_2sigma_density = math.ceil(sm_unrolling * (op_pres + idx_precision) * (average_density + 2 * density_std))
    bw_3sigma_density = math.ceil(sm_unrolling * (op_pres + idx_precision) * (average_density + 3 * density_std))
    bw_6sigma_density = math.ceil(sm_unrolling * (op_pres + idx_precision) * (average_density + 6 * density_std))
    bw_dense = math.ceil(sm_unrolling * (op_pres + idx_precision))
    logging.warning(
        f"bw_aver_density: {bw_aver_density}, bw_1sigma: {bw_1sigma_density}, bw_2sigma: {bw_2sigma_density},"
        f"bw_3sigma: {bw_3sigma_density}, bw_6sigma: {bw_6sigma_density}, bw_dense: {bw_dense}")
    bw_pool = sorted(
        [bw_aver_density, bw_1sigma_density, bw_2sigma_density, bw_3sigma_density, bw_dense, int(bw_dense * 1.2)])
    return bw_pool


def plot_datapath(config_collect, pe_pool,
                  lat_datapath_collect_mu,
                  lat_datapath_collect_std,
                  ee_datapath_collect_mu,
                  ee_datapath_collect_std):
    # Data organized by configurations
    configs = {
        ('gating', 'gating'): {
            'pe_count': pe_pool,
            'util_mu': [],
            'util_std': [],
            'lat_mu': [],
            'lat_std': [],
            'ee_mu': [],
            'ee_std': []
        },
        ('gating', 'skipping'): {
            'pe_count': pe_pool,
            'util_mu': [],
            'util_std': [],
            'lat_mu': [],
            'lat_std': [],
            'ee_mu': [],
            'ee_std': []
        },
        ('skipping', 'gating'): {
            'pe_count': pe_pool,
            'util_mu': [],
            'util_std': [],
            'lat_mu': [],
            'lat_std': [],
            'ee_mu': [],
            'ee_std': []
        },
        ('skipping', 'skipping'): {
            'pe_count': pe_pool,
            'util_mu': [],
            'util_std': [],
            'lat_mu': [],
            'lat_std': [],
            'ee_mu': [],
            'ee_std': []
        }
    }
    for saf_config_index in range(len(config_collect)):
        config = config_collect[saf_config_index]
        sample_count = len(pe_count_collect)
        configs[config]['lat_mu'] = lat_datapath_collect_mu[
                                    sample_count * saf_config_index:sample_count * (saf_config_index + 1)]
        configs[config]['lat_std'] = lat_datapath_collect_std[
                                     sample_count * saf_config_index:sample_count * (saf_config_index + 1)]
        configs[config]['ee_mu'] = ee_datapath_collect_mu[
                                   sample_count * saf_config_index:sample_count * (saf_config_index + 1)]
        configs[config]['ee_std'] = ee_datapath_collect_std[
                                    sample_count * saf_config_index:sample_count * (saf_config_index + 1)]

    # Create figure with subplots
    # plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5))

    # Colors and markers for different configurations
    gating_color = 'green'  # yellow
    skipping_color = u'#b32828'  # red 'purple'
    styles = {
        ('gating', 'gating'): {'color': '#4B88B5', 'marker': 'D', 'label': 'gating-gating'},
        ('gating', 'skipping'): {'color': 'green', 'marker': 'o', 'label': 'gating-skipping'},
        ('skipping', 'gating'): {'color': '#D17575', 'marker': '^', 'label': 'skipping-gating'},
        ('skipping', 'skipping'): {'color': u'#b32828', 'marker': 's', 'label': 'skipping-skipping'}
    }

    labels = {
        ('gating', 'gating'): 'CG-A/CG-W',
        ('gating', 'skipping'): 'CG-A/SK-W',
        ('skipping', 'gating'): 'SK-A/CG-W',
        ('skipping', 'skipping'): 'SK-A/SK-W'
    }

    # Common error bar settings
    error_bar_props = {
        'capsize': 5,
        'capthick': 2,
        'elinewidth': 2,
        'alpha': 1
    }

    # Plot 1: Latency vs PE Count
    # ax1.set_title('Latency vs PE Count', pad=15, fontsize=12)
    ax1.set_xlabel('PE Count', fontsize=10)
    ax1.set_ylabel('Latency [cc]', fontsize=10, weight='bold')

    pe_size_vec = [x + 1 for x in range(len(configs[config]['pe_count']))]
    for saf_config_index in range(len(config_collect)):
        config = config_collect[saf_config_index]
        style = styles[config]
        # ax1.errorbar(pe_size_vec, configs[config]['lat_mu'],
        #              yerr=configs[config]['lat_std'],
        #              color=style['color'], marker=style['marker'],
        #              label=labels[config], markersize=5,
        #              **error_bar_props)
        configs[config]['lat_mu'] = np.array(configs[config]['lat_mu'])
        configs[config]['lat_std'] = np.array(configs[config]['lat_std'])
        ax1.plot(pe_size_vec, configs[config]['lat_mu'], color=style['color'], marker=style['marker'],
                 label=labels[config], markersize=6, markeredgecolor='w', markeredgewidth=1)
        ax1.fill_between(pe_size_vec, np.maximum(0, configs[config]['lat_mu'] - 3 * configs[config]['lat_std']),
                         configs[config]['lat_mu'] + 3 * configs[config]['lat_std'],
                         alpha=0.3, color=style['color'])

    ax1.grid(True, alpha=0.8)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_xticks(pe_size_vec,
                   [x1 * x2 for (x1, x2) in pe_pool],
                   rotation=0,
                   ha='center',
                   fontsize=10)
    ax1.set_yscale('log')

    # Plot 2: Energy vs PE Count
    # ax2.set_title('Energy vs PE Count', pad=15, fontsize=12)
    ax2.set_xlabel('PE Count', fontsize=10)
    ax2.set_ylabel('Energy [pJ]', fontsize=10, weight='bold')

    for saf_config_index in range(len(config_collect)):
        config = config_collect[saf_config_index]
        style = styles[config]
        # ax2.errorbar(pe_size_vec, configs[config]['ee_mu'],
        #              yerr=configs[config]['ee_std'],
        #              color=style['color'], marker=style['marker'],
        #              label=labels[config], markersize=5,
        #              **error_bar_props)
        configs[config]['ee_mu'] = np.array(configs[config]['ee_mu'])
        configs[config]['ee_std'] = np.array(configs[config]['ee_std'])
        ax2.plot(pe_size_vec, configs[config]['ee_mu'], color=style['color'], marker=style['marker'],
                 label=labels[config], markersize=6, markeredgecolor='w', markeredgewidth=1)
        ax2.fill_between(pe_size_vec, np.maximum(0, configs[config]['ee_mu'] - 3 * configs[config]['ee_std']),
                         configs[config]['ee_mu'] + 3 * configs[config]['ee_std'],
                         alpha=0.3, color=style['color'])

    ax2.grid(True, alpha=0.8)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.set_xticks(pe_size_vec,
                   [x1 * x2 for (x1, x2) in pe_pool],
                   rotation=0,
                   ha='center',
                   fontsize=10)
    ax2.set_yscale('log')

    # Set x-axis to log scale for all plots since PE count varies exponentially
    # ax1.set_xscale('log', base=2)
    # ax2.set_xscale('log', base=2)

    # Adjust layout and spacing
    plt.tight_layout()
    plt.show()


def plot_mem(bw, lat_mu_gating, lat_std_gating, ee_mu_gating, ee_std_gating,
             lat_mu_skipping, lat_std_skipping, ee_mu_skipping, ee_std_skipping):
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 3), gridspec_kw={'width_ratios': [3, 1]})
    # fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
    # gating_color = '#4B88B5'  # Soft blue
    # skipping_color = '#D17575'  # Soft red
    gating_color = 'green'  # yellow
    skipping_color = u'#b32828'  # red

    # Plot latency data
    lat_mu_gating = np.array(lat_mu_gating)
    lat_std_gating = np.array(lat_std_gating)
    lat_mu_skipping = np.array(lat_mu_skipping)
    lat_std_skipping = np.array(lat_std_skipping)
    x_vec = bw
    x_vec = np.array(['$\mu ll_{sp}$', '$\sigma ll_{sp}$', '2$\sigma ll_{sp}$', '3$\sigma ll_{sp}$', 'dense', '>dense'])
    ax1.plot(x_vec, lat_mu_gating, '-', color=gating_color, marker='o',
             label='CG-A/SK-W', markersize=6, linewidth=2, markeredgecolor='w', markeredgewidth=1)
    ax1.fill_between(x_vec, np.maximum(0, lat_mu_gating - 3 * lat_std_gating),
                     lat_mu_gating + 3 * lat_std_gating,
                     alpha=0.3, color=gating_color)
    ax1.plot(x_vec, lat_mu_skipping, '-', color=skipping_color, marker='s',
             label='SK-A/SK-W', markersize=6, linewidth=2, markeredgewidth=1, markeredgecolor='white')
    ax1.fill_between(x_vec, np.maximum(0, lat_mu_skipping - 3 * lat_std_skipping),
                     lat_mu_skipping + 3 * lat_std_skipping,
                     alpha=0.3, color=skipping_color)
    ax1.set_xticklabels(x_vec, rotation=30, ha='center')
    ax1.set_xlabel('Bandwidth', fontsize=12, weight='normal')
    ax1.set_ylabel('Latency [cc]', fontsize=12, weight='normal')
    # ax1.set_title('Latency/Energy vs Bandwidth', fontsize=12, weight='bold')
    ax1.grid(True)
    ax1.set_axisbelow(True)
    ax1.legend(loc='lower left')

    # Plot energy efficiency data
    ee_mu_gating = np.array(ee_mu_gating)
    ee_std_gating = np.array(ee_std_gating)
    ee_mu_skipping = np.array(ee_mu_skipping)
    ee_std_skipping = np.array(ee_std_skipping)

    # ax2 will be in bar, as the energy is constant with the memory bandwidth
    scheme_vec = np.array(["CG-A/SK-W", "SK-A/SK-W"])
    lat_mu_vec = np.array([ee_mu_gating[0], ee_mu_skipping[1]])
    lat_std_vec = np.array([ee_std_gating[0], ee_std_skipping[0]])
    ax2.bar(scheme_vec, lat_mu_vec, color=[gating_color, skipping_color], edgecolor='black', width=0.5)
    ax2.set_xticklabels(scheme_vec, rotation=30, ha='center')

    # ax2.plot(bw, ee_mu_gating, '-', color=gating_color, marker='o',
    #          label='CG-A/SK-W', markersize=6, linewidth=2, markeredgecolor='w', markeredgewidth=1)
    # ax2.fill_between(bw, ee_mu_gating - 3 * ee_std_gating,
    #                  ee_mu_gating + 3 * ee_std_gating,
    #                  alpha=0.3, color=gating_color)
    # ax2.plot(bw, ee_mu_skipping, '-', color=skipping_color, marker='s',
    #          label='SK-A/SK-W', markersize=6, linewidth=2, markeredgewidth=1, markeredgecolor='white')
    # ax2.fill_between(bw, ee_mu_skipping - 3 * ee_std_skipping,
    #                  ee_mu_skipping + 3 * ee_std_skipping,
    #                  alpha=0.3, color=skipping_color)
    #
    # ax2.set_xlabel('Bandwidth', fontsize=12, weight='normal')
    ax2.set_ylabel('Energy [pJ]', fontsize=12, weight='normal')
    # ax2.grid(True)
    # ax2.set_axisbelow(True)
    # ax2.legend(loc='upper right')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plot
    plt.show()


def get_layer_shape_and_bw(model_name: str = "resnet18", layer_id: int = 2,
                  d2_dim_size: float = 8, debug: bool = False):
    """ return layer shape info for a layer, and the bandwidth corresponding to the average/3std/dense requirement """
    """ @param debug: used for exp3 """
    dataset_name = "imagenet"
    tile_size: dict = {"I": 8, "W": 8}
    pkl_layer_shape = f"../../zigzag/density_parser/pkl/layer_shape/{model_name}/shape_{model_name}.pkl"
    with open(pkl_layer_shape, "rb") as fp:
        workload = pickle.load(fp)[layer_id]
    # read in average weight density
    pkl_weight = f"../../zigzag/density_parser/pkl/weight/{model_name}/" \
                 f"dist_{model_name}_layer{layer_id}_tile{tile_size['W']}.pkl"
    try:
        with open(pkl_weight, "rb") as fp:
            con: list = pickle.load(fp)
            spar_weight: dict = {
                "density_list": con[0],
                "density_occurrence": con[1],
                "density_mean": con[2],
                "density_std": con[3],
            }
        weight_density = spar_weight["density_mean"]
    except FileNotFoundError:  # for resnet18, sparse network from Man miss some layers
        weight_density = 1
    # read in average act density, act std
    if debug:
        pkl_act = f"../../zigzag/density_parser/pkl/act_debug/{dataset_name}/{model_name}/" \
                  f"dist_{model_name}_{dataset_name}_layer{layer_id}_tile{tile_size['I']}.pkl"
    else:
        pkl_act = f"../../zigzag/density_parser/pkl/act/{dataset_name}/{model_name}/" \
                  f"dist_{model_name}_{dataset_name}_layer{layer_id}_tile{tile_size['I']}.pkl"
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
    act_average_density = np.mean(spar_act["density_mean_collect"])
    act_density_std = np.std(spar_act["density_mean_collect"])
    sm_unrolling = min(workload["C"] * act_average_density * weight_density, d2_dim_size)
    if sm_unrolling != d2_dim_size:
        logging.warning(f"The d2 (C) dim is not fully occupied {sm_unrolling}/{d2_dim_size}")
    bw_pool = derive_6_sigma_bw(ox=workload["OX"], oy=workload["OY"], c=workload["C"],
                                weight_density=weight_density, average_density=act_average_density,
                                density_std=act_density_std, sm_unrolling=sm_unrolling,
                                encoding="bm", tile_size=8, op_pres=8)
    return workload, bw_pool


if __name__ == "__main__":
    """
    Exp: system performance @ L2, ResNet18
    """
    logging_level = logging.WARNING  # logging level
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)

    """ Exp1 setting """
    # layer_id = 26
    # model_name = "resnet50"
    # saf_pool = [("gating", "skipping"), ("skipping", "skipping")]  # exp parameters
    # pe_pool = [(32, 32)]
    # workload, bw_pool = get_layer_shape_and_bw(model_name=model_name, layer_id=layer_id, d2_dim_size=pe_pool[0][1])

    """ Exp2 setting """
    layer_id = 26
    model_name = "resnet50"
    saf_pool = [("skipping", "skipping")]  # exp parameters
    workload, __ = get_layer_shape_and_bw(model_name=model_name, layer_id=layer_id)
    bw_pool = [8700]  # exp parameters (bit)
    pe_pool = [(32, 32)]

    plot_func = "1"  # initialization
    if len(bw_pool) > 1:
        assert len(pe_pool) == 1
        plot_func = "1"
    if len(pe_pool) > 1:
        assert len(bw_pool) == 1
        plot_func = "2"

    # Exp3
    # means = np.array([110804, 94413, 81407, 70914, 62327])
    # stds = np.array([17979, 14140, 12192, 10621, 9334])
    # freq = 500  # MHz
    # # convert to time ms
    # means = means / (freq * 1e3)
    # stds = stds / (freq * 1e3)
    # labels = ["(24, 24) PE (failed)", "(26, 26) PE (50%)", "(28, 28) PE (68%)", "(30, 30) PE (96%)", "(32, 32) PE (99.6%)"]
    # plot_multiple_gaussians(means=means, stds=stds, labels=labels)
    # exit()

    lat_mu_gating = []
    lat_std_gating = []
    ee_mu_gating = []
    ee_std_gating = []
    lat_mu_skipping = []
    lat_std_skipping = []
    ee_mu_skipping = []
    ee_std_skipping = []
    lat_mu_together = []
    lat_std_together = []
    ee_mu_together = []
    ee_std_together = []
    pe_count_collect = [pe_pair[0] * pe_pair[1] for pe_pair in pe_pool]
    for saf_pair in saf_pool:
        act_saf, weight_saf = saf_pair
        assert act_saf in ["gating", "skipping"]
        assert weight_saf in ["gating", "skipping"]
        for mem_bw in bw_pool:
            for pe_pair in pe_pool:
                act_r_cost, act_w_cost = get_memory_cost(mem_size_in_byte=512 * 1024, mem_bw_in_bit=mem_bw)
                # act_r_cost = r_costs[mem_bw]
                # act_w_cost = w_costs[mem_bw]
                exp = exp_sigma(act_mem_bw=mem_bw, act_saf=act_saf, act_r_cost=act_r_cost, act_w_cost=act_w_cost,
                                weight_saf=weight_saf, pe_pair=pe_pair, workload=workload, layer_id=layer_id,
                                model_name=model_name)
                total_lats, total_lats_std, total_ees, total_ees_std = exp.simulation()
                if act_saf == "gating":
                    lat_mu_gating.append(total_lats)
                    lat_std_gating.append(total_lats_std)
                    ee_mu_gating.append(total_ees)
                    ee_std_gating.append(total_ees_std)
                else:
                    lat_mu_skipping.append(total_lats)
                    lat_std_skipping.append(total_lats_std)
                    ee_mu_skipping.append(total_ees)
                    ee_std_skipping.append(total_ees_std)
                lat_mu_together.append(total_lats)
                lat_std_together.append(total_lats_std)
                ee_mu_together.append(total_ees)
                ee_std_together.append(total_ees_std)
                logging.warning(
                    f"saf: {saf_pair}, pe_pair: {pe_pair}, bw: {mem_bw}, lat_cc: {total_lats}, lat_std: {total_lats_std}, 3lat_std/lat_cc: {3 * total_lats_std / total_lats}, "
                    f"ee_cc: {total_ees}, ee_std: {total_ees_std}, 3ee_std/ee_cc: {3 * total_ees_std / total_ees}")
    if plot_func == "1":
        plot_mem(bw_pool, lat_mu_gating, lat_std_gating, ee_mu_gating, ee_std_gating,
                 lat_mu_skipping, lat_std_skipping, ee_mu_skipping, ee_std_skipping)
    else:
        plot_datapath(saf_pool, pe_pool,
                      lat_mu_together,
                      lat_std_together,
                      ee_mu_together,
                      ee_std_together)
