
import matplotlib.pyplot as plt
import numpy as np

# This file is for model validation. For details, please refer to the comments under the main program.
# Each function can be run as a standalone function and does not call any external function.


def sparseloop_validation_single_pe():
    # This function is for validation to Sparseloop on single pe behavior The hardware template is the one used in
    # Fig. 12 in Sparseloop paper. The energy, area value is collected from their code at:
    # https://github.com/Accelergy-Project/micro22-sparseloop-artifact/tree/main/workspace/2022.micro.artifact
    # /evaluation_setups/fig12_eyerissv2_pe_setup

    def plot_bar_chart(layers: list, y1: list, y2: list, y3: list, x_label: str = None, y_label: str = None, title: str = None, log_x=False, log_y=False, width=0.4):
        # Plotting the bar chart with dark border
        x = np.arange(len(layers))
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
        axs.bar(x - width/2, y1, edgecolor='black', color='#ccd5ae', width=width, label="Eyeriss V2")
        # axs.bar(x, y2, edgecolor='black', width=width, label="Sparseloop", hatch="-")
        axs.bar(x + width/2, y3, edgecolor='black', color='#fff6d5', width=width, label="SunPar")

        plt.xticks(range(len(layers)), layers)

        # log x
        if log_x == True:
            x_scale = "log"
            axs.set_xscale(x_scale)

        # log y
        if log_y == True:
            y_scale = "log"
            axs.set_yscale(y_scale)

        # Adding text labels to show mismatch (y3 in terms of y1)
        mm = [(y3[i]/y1[i]-1)*100 for i in range(len(y3))]  # %
        for i, v in enumerate(mm):
            plt.text(i, y3[i]*1.06, f"{round(mm[i], 1)}%", ha="center", va="bottom", fontsize=10, weight="normal",
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))


        # Adding labels and title
        plt.tick_params(axis="x", labelsize=12)
        plt.tick_params(axis="y", labelsize=12)
        axs.set_xlabel(x_label, fontsize=14)
        axs.set_ylabel(y_label, fontsize=14)
        axs.set_title(title)
        plt.legend(fontsize=12, loc="lower right")
        plt.tight_layout()
        # Displaying the chart
        plt.show()

    def plot_mismatch_curve(layers: list, y1: list, y2: list, x_label: str = None, y_label: str = None, title: str = None, log_x=False, log_y=False):
        x = np.arange(len(layers))
        colors = ["gray", "chocolate"]
        plt.plot(x, y1, linestyle="--", color=colors[0], marker="o", markeredgewidth=1, markeredgecolor="black", label="baseline")
        plt.plot(x, y2, linestyle="--", color=colors[1], marker="s", markeredgewidth=1, markeredgecolor="black",
                 label="Sparseloop")
        plt.xticks(range(len(layers)), layers)
        plt.ylim([-10, 10])
        # log x
        if log_x == True:
            x_scale = "log"
            plt.xscale(x_scale)

        # log y
        if log_y == True:
            y_scale = "log"
            plt.yscale(y_scale)

        # Adding labels and title
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        # Displaying the chart
        plt.show()

    ### main ###
    # The method is getting latency value for a dense workload
    # Then scale the latency according to the density of I, W

    # record the latency value
    simulation = False  # I have store the simulated value below. Turn this label to True if want to refetch the result.
    lats = [1592157.7984, 1478911.5903999999, 1114007.1424, 1407188.992, 1610612.736, 1790967.808, 926941.1840000001, 729808.896]
    # Layer order: L07, L09, L13, L19, L21, L23, L25, L27
    ref_lats_sparseloop = [1592245, 1479114, 1114139, 1407304, 1610668, 1791135, 927185, 729915]
    ref_lats_grouptruch = [1702140, 1499682, 1144158, 1457458, 1677684, 1821557, 948572, 775316]

    density_info = {
        "L07": {"I": 0.73, "W": 0.52},
        "L09": {"I": 0.86, "W": 0.82},
        "L13": {"I": 0.83, "W": 0.64},
        "L19": {"I": 0.61, "W": 0.55},
        "L21": {"I": 0.64, "W": 0.60},
        "L23": {"I": 0.61, "W": 0.70},
        "L25": {"I": 0.68, "W": 0.65},
        "L27": {"I": 0.58, "W": 0.30}
    }
    layers = list(density_info.keys())

    # calc mismatch regarding sparseloop result and golden reference (provided by Sparseloop)
    # mismatch_sparseloop = []
    # mismatch_actualdata = []
    # for i in range(len(lats)):
    #     diff = round(lats[i] / ref_lats_sparseloop[i] - 1, 4) * 100  # unit: %
    #     mismatch_sparseloop.append(diff)
    #     diff = round(lats[i] / ref_lats_grouptruch[i] - 1, 4) * 100  # unit: %
    #     mismatch_actualdata.append(diff)

    plot_bar_chart(layers, ref_lats_grouptruch, ref_lats_sparseloop, lats,
                   x_label="MobileNet-V2 Layers",
                   y_label="Latency [cc]")
    # plot_mismatch_curve(layers, mismatch_actualdata, mismatch_sparseloop,
    #                     x_label="MobileNet layers",
    #                     y_label="Mismatch [%]")

def sparseloop_validation_single_mem():
    # This function is to validate the single memory behavior between Sparseloop and ZigZag.
    # Specifically, the latency model used by Sprseloop is based on throttling calculation.
    # the latency model used by zigzag does not involve this calculation.
    # This function is to verify these two methods will have the same result.

    # the used sram is referred from the "Buffer" used by Sparseloop (Fig. 13) in their DSTC hardware template.
    # Reference link: https://github.com/Accelergy-Project/micro22-sparseloop-artifact/blob/main/workspace/2022.micro.artifact/evaluation_setups/fig13_dstc_setup/input_specs/architecture.yaml
    workload_dense = 4096 ** 3
    density_a =[0.3, 0.5, 0.7, 0.9, 1.0]
    density_b = [0.4, 1.0]
    bw = 116
    utilized_pes = {
        "a0.3b0.4": 71.39,
        "a0.5b0.4": 79.99,
        "a0.7b0.4": 84.39,
        "a0.9b0.4": 86.52,
        "a1.0b0.4": 95.83,
        "a0.3b1.0": 95.36,
        "a0.5b1.0": 106.84,
        "a0.7b1.0": 112.72,
        "a0.9b1.0": 115.57,
        "a1.0b1.0": 128,
    }
    dstc_latency_ratio = {  # from https://github.com/Accelergy-Project/micro22-sparseloop-artifact/blob/main/workspace/2022.micro.artifact/evaluation_setups/fig13_dstc_setup/scripts/baseline.yaml
        "a0.3b0.4": 1100,
        "a0.5b0.4": 1480,
        "a0.7b0.4": 1820,
        "a0.9b0.4": 2300,
        "a1.0b0.4": 2500,
        "a0.3b1.0": 1930,
        "a0.5b1.0": 2690,
        "a0.7b1.0": 3300,
        "a0.9b1.0": 4160,
        "a1.0b1.0": 4600,
    }

    def plot_bar_chart(cases: list, y1: list, y2: list, x_label: str = None, y_label: str = None, title: str = None, log_x=False, log_y=False, width=0.4):
        # Plotting the bar chart with dark border
        x = np.arange(len(cases))
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
        colors = [u"#27e1c1", u"#fff6d5"]  # 2rd and 3rd color in default matplotlib colors
        axs.bar(x - width/2, y1, color=colors[0], edgecolor='black', width=width, label="DSTC")
        axs.bar(x + width/2, y2, color=colors[1], edgecolor='black', width=width, label="SunPar")

        plt.xticks(range(len(cases)), cases, rotation=30)

        # log x
        if log_x == True:
            x_scale = "log"
            axs.set_xscale(x_scale)

        # log y
        if log_y == True:
            y_scale = "log"
            axs.set_yscale(y_scale)

        # Adding labels and title
        plt.tick_params(axis="x", labelsize=12)
        plt.tick_params(axis="y", labelsize=12)
        axs.set_xlabel(x_label, fontsize=14)
        axs.set_ylabel(y_label, fontsize=14)
        axs.set_title(title)
        plt.legend(fontsize=12)
        plt.tight_layout()
        # Displaying the chart
        plt.show()


    def calc_lat_sparseloop(da, db, utilized_pe, bw):
        throttling = bw / utilized_pe
        lat = workload_dense * da * db / utilized_pe / throttling
        return lat
    def calc_lat_zigzag(da, db, bw):
        lat = workload_dense * da * db / bw
        return lat
    def calc_lat_reference(da, db, dense_lat):
        case = f"a{da}b{db}"
        dense_case = f"a1.0b1.0"
        lat = dstc_latency_ratio[case]/dstc_latency_ratio[dense_case] * dense_lat
        return lat

    ### main ###
    cases = []
    lats_sparseloop = []
    lats_zigzag = []
    lats_reference = []
    dense_lat = calc_lat_zigzag(1, 1, bw)
    for da in density_a:
        for db in density_b:
            cases.append(f"{da}-{db}")
            utilized_pe = utilized_pes[f"a{da}b{db}"]
            lat_sparseloop = calc_lat_sparseloop(da, db, utilized_pe, bw)
            lats_sparseloop.append(lat_sparseloop)
            lat_zigzag = calc_lat_zigzag(da, db, bw)
            lats_zigzag.append(lat_zigzag)
            lat_reference = calc_lat_reference(da, db, dense_lat)
            lats_reference.append(lat_reference)
    # calc mismatch
    mismatch = []
    for i in range(len(cases)):
        diff = round(lats_zigzag[i] / lats_sparseloop[i] - 1, 4) * 100  # unit: %
        mismatch.append(diff)

    # plot
    plot_bar_chart(cases, lats_sparseloop, lats_zigzag,
                   x_label="A Density - B Density",
                   y_label="Latency [cc]")


if __name__ == "__main__":
    # Following validation cases are included within this file:
    # 1. single pe behavior - func: sparseloop_validation_single_pe()
    #       - In this validation, we will validate our model against sparseloop using the same hardware template
    #       they use in Fig. 12 in their paper, which named "Eyeriss v2". The hardware template actually only include
    #       1 PE element, so it should be shrinked version of Eyeriss v2.
    #       - In the hardware template, the memory bandwidth is sufficient for each memory so that there is no memory
    #       stalling. Therefore, for a dense workload, the total cycle count = #MACs in a layer.
    #       - For sparse workload, the formula I used is: cycle count = #MACs(dense) * density(input) * density(weight)
    #       - Not sure what formula is used within Sparseloop. The validation results shows 0.3% mismatch at maximum.
    #       - The workload used is the one used in their code for Fig. 12, which includes 8 layers of mobilenet0.5.
    #       - The density value is collected from their code.
    #       - Their mapping is not used. Instead, I let the zigzag mapper generate itself. Since there is no stalling,
    #       it will affect the final latency result.
    #       - The energy and area value is not compared in this experiment.
    # sparseloop_validation_single_pe()
    # 2. single mem behavior - func: sparseloop_validation_single_mem()
    #       - In this validation, we will compare if the memory latency model used by Sparseloop and ZigZag is the same.
    #       - The hardware template is DSTC, the one they used for Fig. 13 in Sparseloop paper.
    #       - I did not implement entire hardware in ZigZag, because skipping technique is used in their code, which
    #       affects the average utilized #PEs. Instead, I collect the utilized #PEs from their output. Then calculate
    #       the latency for the memory.
    #       - The memory I focused on is the one they used for output storage, named "Buffer" in their code.
    #       - To show the memory behavior, I only consider the memory latency, no matter if the compute cycle count
    #       will cover the memory latency or not.
    #       - The result shows 100% match.
    #       - I guess the formula they used is: cycle count = data size / unconstrained_bw / throttling
    #       in which, throttling = real_bw / unconstrained_bw
    #       - The one used by zigzag is: cycle count = data size / real_bw
    sparseloop_validation_single_mem()