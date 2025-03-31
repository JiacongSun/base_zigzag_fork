import sys
sys.path.append('/users/micas/jsun/sunjc/codes/intel_project/sunpar/base_zigzag_fork/')
import copy
from exp_mem_util import derive_idx_precision
import math
import numpy as np
from sigma_like import exp_sigma
import logging
import matplotlib.pyplot as plt
import pickle
from exp_sys_lat_ee import get_memory_cost, get_layer_shape_and_bw
import pandas as pd
import time



def derive_model_layer_count(model_name: str):
    # Class NetworkInference in inference_hooker.py has its own function to derive this info
    layer_count_dict: dict = {
        "resnet18": 21,
        "resnet18_sparse": 17,  # does not have downsample layers (4) and fc layer (1)
        "resnet50": 54,  # 54 in the hooker, but 53 in zigzag onnx
        "vgg19": 19,
        "mobilenetv3": 54,
        "mobilenetv2": 53,
        "mobilenetv2_sparse": 50,  # see comments of resnet18_sparse
        "quant_mobilenetv2": 53,
        "efficientnetb0_sparse": 49,  # see comments of resnet18_sparse
    }
    return layer_count_dict[model_name]


def get_layerwise_covariance_matrix(dataset_name, model_name, tile_size_i: int = 8, debug: bool = False):
    if debug:
        filename = f"../../zigzag/density_parser/pkl/layerwise_cm_debug/{dataset_name}/cm_{model_name}_tile{tile_size_i}.pkl"
    else:
        filename = f"../../zigzag/density_parser/pkl/layerwise_cm/{dataset_name}/cm_{model_name}_tile{tile_size_i}.pkl"
    with open(filename, "rb") as fp:
        load = pickle.load(fp)
    return load


def get_inference_perf(saf_pair, mem_bw, mem_size_kb, pe_pair, freq, pe_area, model_name, dataset_name, debug=False,
                       enable_double_buffer=True):
    """
    get performance of the entire inference
    :param saf_pair: (act, w), including ("gating", "gating"), ("gating", "skipping"), ("skipping", "skipping")
    :param mem_bw: int (bit)
    :param mem_size_kb: float
    :param pe_pair: tuple, e.g. (32, 32)
    :param freq: hardware frequency (MHz)
    :param pe_area: area per PE (mm2)
    :param model_name: model name
    :param dataset_name: dataset name
    :param enable_double_buffer: if enable double buffer when calc dram latency mu and std
    """
    act_saf, weight_saf = saf_pair
    assert act_saf in ["gating", "skipping"]
    assert weight_saf in ["gating", "skipping"]
    # calc on-chip input buffer size (KB) to cover peak throughput under the worst case (ref to the buf in TrapeZoid)
    if saf_pair == ("skipping", "skipping"):
        op_precision = 8
        buffer_size_kb = np.prod(pe_pair) * op_precision / 8 / 1024
    else:
        buffer_size_kb = 0
    layer_count = derive_model_layer_count(model_name=model_name)
    act_r_cost, act_w_cost, mem_area = get_memory_cost(mem_size_in_byte=mem_size_kb * 1024,
                                                       mem_bw_in_bit=mem_bw,
                                                       return_area=True)
    sys_area = round(mem_area * 2 + pe_area[saf_pair] * np.prod(pe_pair), 3)
    layer_lat_mu_collect = []
    layer_lat_std_collect = []
    layer_lat_scale_collect = []
    layer_ee_mu_collect = []
    layer_ee_std_collect = []
    layer_ee_scale_collect = []
    # read in layer-wise cm
    layerwise_cm: np.ndarray = get_layerwise_covariance_matrix(model_name=model_name,
                                                               dataset_name=dataset_name,
                                                               tile_size_i=8,
                                                               debug=debug)
    for layer_id in range(layer_count):
        # get layer-wise perf
        workload, bw_req_collect = get_layer_shape_and_bw(model_name=model_name, layer_id=layer_id, debug=debug)
        exp = exp_sigma(act_mem_bw=mem_bw, act_saf=act_saf, act_r_cost=act_r_cost,
                        act_w_cost=act_w_cost,
                        weight_saf=weight_saf, pe_pair=pe_pair, workload=workload, layer_id=layer_id,
                        model_name=model_name, act_mem_size=mem_size_kb * 1024 * 8)
        total_lats, total_lats_std, total_ees, total_ees_std = exp.simulation(debug=debug, enable_double_buffer=enable_double_buffer)
        mem_bottleneck_id = exp.return_mem_bottleneck_id()
        lat_scaling_factor, ee_scaling_factor = exp.return_scaling_factor()
        layer_lat_mu_collect.append(total_lats)
        layer_lat_std_collect.append(total_lats_std)
        layer_lat_scale_collect.append(lat_scaling_factor)
        layer_ee_mu_collect.append(total_ees)
        layer_ee_std_collect.append(total_ees_std)
        layer_ee_scale_collect.append(ee_scaling_factor)
        # print(mem_bottleneck_id)
    """ calc inf-wise perf """
    # pre-processing
    layer_lat_scale_collect_np = np.array(layer_lat_scale_collect)
    layer_ee_scale_collect_np = np.array(layer_ee_scale_collect)
    inf_lat_var = layer_lat_scale_collect_np @ layerwise_cm @ layer_lat_scale_collect_np.T
    inf_ee_var = layer_ee_scale_collect_np @ layerwise_cm @ layer_ee_scale_collect_np.T
    # calc results
    inf_lat_mu = sum(layer_lat_mu_collect)
    inf_lat_ee = sum(layer_ee_mu_collect)
    inf_lat_std = inf_lat_var ** 0.5
    inf_ee_std = inf_ee_var ** 0.5
    inf_lat_std_naive = sum(layer_lat_std_collect)  # pure sum
    """ output results """
    # note for mem_act: the size includes both act and output, so it doubles
    time_mu = round(inf_lat_mu * 1000 / freq / 1e6, 2)
    time_std = round(inf_lat_std * 1000 / freq / 1e6, 2)
    three_std_time = round(time_mu + 3 * time_std, 2)
    if three_std_time < time_mu:
        breakpoint()
    """
    why at least mem_size_kb * 2: this is to cover both input and output on-chip memory, assumed with the same size
    """
    if enable_double_buffer:
        mem_scale_factor = 4  # half of mem is saved to support double buffering
    else:
        mem_scale_factor = 2
    act_mem_size_kb = mem_size_kb * mem_scale_factor
    real_act_mem_size_kb = act_mem_size_kb + buffer_size_kb
    logging.critical(
        f"saf(act, w): {saf_pair}, mem_act(kb): {real_act_mem_size_kb} ({act_mem_size_kb}), bw(bit): {mem_bw}, pe pair: {pe_pair}, area(mm2): {sys_area}, time mu: {time_mu} ms, 3-sigma time: {three_std_time} ms, abs 3-sigma time: {round(3 * time_std, 2)} ms, rela 3-sigma/mu: {round(3 * time_std / time_mu, 2)}")
    return act_mem_size_kb, real_act_mem_size_kb, sys_area, time_mu, three_std_time


def plot_dse_results(df: pd.DataFrame, threshold: float, x: str, y: str, perf: str, show_std: False):
    """
    plot the results in a scatter plot, x: mem_act, y: mem_bw, color: time (red if > threshold, blue otherwise)
    """
    df[perf] = df[perf].round(1)
    # Normalize time values for color intensity
    time_normalized = (df[perf] - df[perf].min()) / (df[perf].max() - df[perf].min())

    if show_std:
        # Calculate size
        # Normalize the sizes to be reasonable for plotting
        size_values = df['time_three_std'] - df['time_mu']
        # Make all sizes positive and scale them to reasonable dot sizes
        size_scaled = ((size_values - size_values.min()) * 100) + 50

    # Create figure and axis
    plt.figure(figsize=(5, 5))

    # Split data into two groups based on time threshold
    time_over = df[perf] > threshold
    time_under = df[perf] <= threshold

    # Split points with perf > threshold and 3-std < threshold
    ans = df[(df.time_mu <= threshold) & (df.time_three_std > threshold)]

    if show_std:
        # Plot points with perf > threshold and 3-std < threshold in red
        # mask = time_over & three_std_time_under
        plt.scatter(ans[x],
                    ans[y],
                    s=size_scaled.loc[ans.index],  # Set the size
                    label='_nolegend_',  # This prevents this scatter from appearing in legend
                    edgecolors='red',
                    linewidth=5)

        # Plot points with perf > threshold in red
        plt.scatter(df.loc[time_over, x],
                    df.loc[time_over, y],
                    c=time_normalized[time_over],
                    s=size_scaled[time_over],  # Set the size
                    cmap='Reds',
                    label='_nolegend_',  # This prevents this scatter from appearing in legend
                    alpha=0.7)  # Add some transparency for overlapping points

        # Plot points with perf <= threshold in blue
        plt.scatter(df.loc[time_under, x],
                    df.loc[time_under, y],
                    c=time_normalized[time_under],
                    s=size_scaled[time_under],  # Set the size
                    cmap='Blues',
                    label='_nolegend_',
                    alpha=0.7)
        # Add separate scatter points just for legend with fixed size
        plt.scatter([], [], c='red', s=100, label=f'Average failed')
        plt.scatter([], [], c='blue', s=100, label=f'Average passed')
    else:
        # Plot points with perf > threshold in red
        plt.scatter(df.loc[time_over, x],
                    df.loc[time_over, y],
                    c=time_normalized[time_over],
                    cmap='Reds',
                    label=f'Average failed')

        # Plot points with perf <= threshold in blue
        plt.scatter(df.loc[time_under, x],
                    df.loc[time_under, y],
                    c=time_normalized[time_under],
                    cmap='Blues',
                    label=f'Average passed')

    # Add time labels for each point
    for idx, row in df.iterrows():
        plt.annotate(f'{row[perf]:.1f}\n{row["time_three_std"]:.1f}',
                     (row[x], row[y]),
                     xytext=(0, 0),  # 5 points offset
                     textcoords='offset points',  # Use offset for text position
                     fontsize=8)  # Smaller font size for labels

    # Add colorbar
    # plt.colorbar(label='Normalized Latency')

    # Add labels and title
    if x in ['mem_size_kb', 'real_mem_size_kb']:
        plt.xlabel('Memory Size [KB]')
        plt.xscale('log')
    elif x == 'mem_bw':
        plt.xlabel('Memory Bandwidth [bit/access]')
    else:
        plt.xlabel(f'{x}')
    if y == 'pe_count':
        plt.ylabel('PE Count')
    else:
        plt.ylabel(f'{y}')
    # plt.title('Memory Access vs Bandwidth with Time Representation')
    plt.legend(loc='upper right')

    # Show the plot
    plt.tight_layout()
    plt.show()
    # plt.savefig("dse.png",
    #         dpi=300,           # Higher DPI for better resolution
    #         bbox_inches='tight',  # Removes extra whitespace
    #         transparent=False,   # Transparent background
    #         format='png')


def plot_schmoo(df: pd.DataFrame, threshold: float, x: str, y: str, perf: str, show_std: False):
    """
    plot the results in a schmoo plot, color: time (red if > threshold, blue otherwise)
    """
    df[perf] = df[perf].round(1)

    # Define unique grid positions
    x_unique = np.sort(df[x].unique())  # Unique x values
    y_unique = np.sort(df[y].unique())  # Unique y values
    x_grid, y_grid = np.meshgrid(x_unique, y_unique)  # Create grid

    # Create a color matrix based on conditions
    color_matrix = np.full(x_grid.shape, np.nan)  # Fill with NaN initially

    # Create a mapping dictionary from DataFrame to grid
    color_map = {(row[x], row[y]): (row["time_mu"], row["time_three_std"]) for _, row in df.iterrows()}

    # Calculate the 3std
    variation_map = {(row[x], row[y]): max(0, (row["time_three_std"] - row["time_mu"])) for _, row in df.iterrows()}
    mean_map = {(row[x], row[y]): row["time_mu"] for _, row in df.iterrows()}

    # Populate the color matrix
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            x_val = x_grid[i, j]
            y_val = y_grid[i, j]
            if (x_val, y_val) in color_map:
                time_mu, time_three_std = color_map[(x_val, y_val)]
                if time_mu > threshold:
                    color_matrix[i, j] = 2  # Red
                elif time_mu <= threshold < time_three_std:
                    color_matrix[i, j] = 1  # Blue
                else:
                    color_matrix[i, j] = 0  # Green

    # Define the color map
    # cmap = plt.cm.colors.ListedColormap(["green", "blue", "red"])
    # bounds = [-0.5, 0.5, 1.5, 2.5]
    # norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    # Plot using pcolormesh()
    plt.figure(figsize=(5, 3))
    # color_list = [u'#cbc0dd', u'#dddcdc', u'#f1bcbd']  # 3std passed, aver passed, failed
    # cmap = plt.cm.colors.ListedColormap(color_list)
    # bounds = [-0.5, 0.5, 1.5, 2.5]
    # norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    # plt.pcolormesh(np.log2(x_unique), np.sqrt(y_unique), color_matrix, cmap=cmap, norm=norm, edgecolors="k",
    #                linewidth=1, shading='auto')
    plt.pcolormesh(np.log2(x_unique), np.sqrt(y_unique), color_matrix, cmap='coolwarm', edgecolors="k", linewidth=1, shading='auto')
    # plt.xscale("log")

    # Label each dot
    for x in np.log2(x_unique):
        for y in np.sqrt(y_unique):
            x_val = 2 ** x
            y_val = y ** 2
            mean_label = round(mean_map[(x_val, y_val)], 1)
            std_label = round(variation_map[(x_val, y_val)], 1)
            plt.text(x, y, f'{mean_label}\n±{std_label}', ha='center', va='center', fontsize=8, weight='normal')

# Set the new xy ticks with squared labels
    plt.xticks(np.log2(x_unique), labels=[f"{int(val)}" for val in x_unique])
    plt.yticks(np.sqrt(y_unique), labels=[f"{int(val)}" for val in y_unique])

    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color="blue", label=f"Worst passed"),
        Patch(color="gray", label=f"Average passed/worst failed"),
        Patch(color="red", label=f"Average failed"),
    ]
    plt.legend(handles=legend_patches, loc="upper right")
    plt.xlabel("Memory Size [KB]", fontsize=12)
    plt.ylabel("PE Count", fontsize=12)

    # Show the plot
    plt.tight_layout()
    plt.show()
    pass


if __name__ == "__main__":
    """
    Exp: system performance @ L2, ResNet18
    """
    logging_level = logging.CRITICAL  # logging level
    # logging_format = "%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging_format = "%(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)
    """ Exp3 setting """
    model_name = "resnet50"
    dataset_name = "imagenet"
    saf_pool = [("gating", "skipping"), ("skipping", "skipping")]  # exp parameters , ("skipping", "skipping")
    # saf_pool = [("skipping", "skipping")]  # exp parameters , ("skipping", "skipping")
    pe_pool = [(x, x) for x in range(10, 151, 10)]
    bw_pool = [64*x for x in range(1, 17, 1)]  # exp parameters (bit)
    mem_size_pool = [2**x for x in range(0, 15, 1)]  # on-chip act mem size (KB)
    enable_double_buffer = False  # True: dram latency can overlap with on-chip sram; False: cannot overlap
    debug = True  # True: 50% of being pure black inputs (images)
    append_result = False  # True: append results to previous pkl
    only_plot = True  # False: also do simulation
    if debug:
        pkl_filename = "sys_dse_debug.pkl"
    else:
        pkl_filename = "sys_dse.pkl"

    ## TODO: debugging
    assert len(pe_pool) > 0
    assert len(saf_pool) > 0
    assert len(bw_pool) > 0
    assert len(mem_size_pool) > 0
    # TODO: ----

    for debug_sim in [True, False]:
        if debug_sim:
            pkl_filename_save = "sys_dse_debug.pkl"
        else:
            pkl_filename_save = "sys_dse.pkl"
        freq = 500  # MHz
        pe_area: dict = {("gating", "gating"): 0.00072, ("gating", "skipping"): 0.0011, ("skipping", "skipping"): 0.00125}  # extracted from sigma and trapezoid
        """ Experiment details """
        if not only_plot:
            time_A = time.time()
            results = []
            for saf_pair in saf_pool:
                for mem_bw in bw_pool:
                    for mem_size_kb in mem_size_pool:
                        for pe_pair in pe_pool:
                            act_mem_size_kb, real_act_mem_size_kb, sys_area, time_mu, three_std_time = get_inference_perf(saf_pair, mem_bw, mem_size_kb, pe_pair, freq, pe_area, model_name, dataset_name, debug, enable_double_buffer)
                            results.append([saf_pair, mem_bw, act_mem_size_kb, real_act_mem_size_kb, pe_pair, sys_area, time_mu, three_std_time])
            results_in_pd = pd.DataFrame(results, columns=["saf", "mem_bw", "mem_size_kb", "real_mem_size_kb", "pe_pair", "area", "time_mu", "time_three_std"])
            """ save results in pkl """
            if append_result:
                with open(pkl_filename_save, "rb") as fp:
                    results_in_pd_pre = pickle.load(fp)
                results_in_pd = pd.concat([results_in_pd_pre, results_in_pd], ignore_index=True)
            with open(pkl_filename_save, "wb") as fp:
                pickle.dump(results_in_pd, fp)
            pd.set_option("display.max_columns", None)
            pd.set_option('display.max_rows', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            time_B = time.time()
            print(f"Total time (min): {(time_B-time_A)/60}")
    # plot the figure
    with open(pkl_filename, "rb") as fp:
        results_in_pd = pickle.load(fp)
    results_in_pd["pe_count"] = results_in_pd["pe_pair"].apply(lambda x: x[0] * x[1])
    ## plot in circle
    # plot_dse_results(df=results_in_pd[(results_in_pd.saf == ("gating", "skipping")) & (results_in_pd.mem_bw == 1024)
    #                                   & (results_in_pd.mem_size_kb < 2**12)], x="mem_size_kb", y="pe_count",
    #                  threshold=8.333, perf="time_mu", show_std=True)
    ## plot in square
    plot_schmoo(df=results_in_pd[(results_in_pd.saf == ("gating", "skipping")) & (results_in_pd.mem_bw == 1024)
                                      & (results_in_pd.mem_size_kb < 2**12)], x="mem_size_kb", y="pe_count",
                     threshold=8.333, perf="time_mu", show_std=True)
    # (results_in_pd.mem_bw > 500) & (results_in_pd.mem_bw < 800) &
    pass
