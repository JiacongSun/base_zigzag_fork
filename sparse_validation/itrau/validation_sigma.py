import math

# workload gemm settings (Fig. 11: case0-3;)
case_ids = [str(i) for i in range(13, 23)]
# case_ids = ["13"]

cases = {
    "0": {  # Fig. 11.a
        "setting": (256, 256, 2048, 0, 0, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "dense",  # gemm type: dense or sparse
        "reference": 0.775,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 0,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "modeling mismatch": -0.012,  # manual recording
        "passed": True,
    },
    "1": {  # Fig.11.b
        "setting": (1024, 16, 500000, 0, 0, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "dense",  # gemm type: dense or sparse
        "reference": 0.1125,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 0,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "modeling mismatch": 0.008,  # manual recording
        "passed": True,
    },
    "2": {  # Fig.11.c
        "setting": (1024, 16, 500000, 0.8, 0.3, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "sparse",  # gemm type: dense or sparse
        "reference": 0.1125,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 0,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "modeling mismatch": 0.008,  # manual recording
        "passed": True,
    },
    "3": {  # Fig.11.d
        "setting": (1024, 16, 500000, 0.8, 0.3, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "sparse",  # gemm type: dense or sparse
        "reference": 0.1125,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 1,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "modeling mismatch": -0.005,  # manual recording
        "passed": True,
    },
    "4": {  # Fig.12.a.case1
        "setting": (128, 2048, 4096, 0, 0, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "dense",  # gemm type: dense or sparse
        "reference": 1 / 1.0625,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 1,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "modeling mismatch": 0.007,  # manual recording
        "passed": True,
    },
    "5": {  # Fig.12.a.case2
        "setting": (320, 3072, 4096, 0, 0, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "dense",  # gemm type: dense or sparse
        "reference": 1 / 1.21875,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 0,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "modeling mismatch": -0.035,  # manual recording
        "passed": True,
    },
    "6": {  # Fig.12.a.case3
        "setting": (1632, 36548, 1024, 0, 0, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "dense",  # gemm type: dense or sparse
        "reference": 1 / 1.0625,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 0,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "modeling mismatch": -0.005,  # manual recording
        "passed": True,
    },
    "7": {  # Fig.12.a.case4
        "setting": (2048, 4096, 32, 0, 0, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "dense",  # gemm type: dense or sparse
        "reference": 1 / 4,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 1,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "modeling mismatch": -0.028,  # manual recording
        "passed": True,
    },
    "7": {  # Fig.12.a.case5  (same to case_id=1)
        "setting": (1024, 16, 500000, 0, 0, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "dense",  # gemm type: dense or sparse
        "reference": 0.1125,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 0,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "modeling mismatch": -0.008,  # manual recording
        "passed": True,
    },
    "8": {  # Fig.12.a.case6
        "setting": (35, 8457, 2560, 0, 0, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "dense",  # gemm type: dense or sparse
        "reference": 1 / 3.375,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 1,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "modeling mismatch": 0.042,  # manual recording
        "passed": True,
    },
    "9": {  # Fig.12.a.case7
        "setting": (31999, 1024, 84, 0, 0, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "dense",  # gemm type: dense or sparse
        "reference": 1 / 1.3125,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 0,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "modeling mismatch": -0.019,  # manual recording
        "passed": True,
    },
    "10": {  # Fig.12.a.case7
        "setting": (84, 1024, 4096, 0, 0, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "dense",  # gemm type: dense or sparse
        "reference": 1 / 1.625,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 1,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "modeling mismatch": 0.013,  # manual recording
        "passed": True,
    },
    "11": {  # Fig.12.a.case8
        "setting": (2048, 1, 128, 0, 0, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "dense",  # gemm type: dense or sparse
        "reference": 1 / 1.125,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 0,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "modeling mismatch": 0.008,  # manual recording
        "passed": True,
    },
    "12": {  # Fig.12.a.case9
        "setting": (256, 256, 2048, 0, 0, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "dense",  # gemm type: dense or sparse
        "reference": 1 / 1.3125,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 0,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "modeling mismatch": 0.005,  # manual recording
        "passed": True,
    },
    "13": {  # Fig.12.b.case1 (green bar)
        "setting": (128, 2048, 4096, 0.8, 0.3, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "sparse",  # gemm type: dense or sparse
        "reference": 1 / 4.5,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 1,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "passed": True,
    },
    "14": {  # Fig.12.b.case2
        "setting": (320, 3072, 4096, 0.8, 0.3, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "sparse",  # gemm type: dense or sparse
        "reference": 1 / 5.5,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 1,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "passed": True,
    },
    "15": {  # Fig.12.b.case3
        "setting": (1632, 36548, 1024, 0.8, 0.3, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "sparse",  # gemm type: dense or sparse
        "reference": 1 / 5,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 1,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "passed": True,
    },
    "16": {  # Fig.12.b.case4
        "setting": (2048, 4096, 32, 0.8, 0.3, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "sparse",  # gemm type: dense or sparse
        "reference": 1 / 16,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 1,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "passed": True,
    },
    "17": {  # Fig.12.b.case5 (same as case id=2)
        "setting": (1024, 16, 500000, 0.8, 0.3, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "sparse",  # gemm type: dense or sparse
        "reference": 0.1125,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 0,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "passed": True,
    },
    "18": {  # Fig.12.b.case6
        "setting": (35, 8457, 2560, 0.8, 0.3, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "sparse",  # gemm type: dense or sparse
        "reference": 1 / 11,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 1,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "passed": False,
    },
    "19": {  # Fig.12.b.case7
        "setting": (31999, 1024, 84, 0.8, 0.3, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "sparse",  # gemm type: dense or sparse
        "reference": 1 / 8,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 1,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "passed": True,
    },
    "20": {  # Fig.12.b.case8
        "setting": (84, 1024, 4096, 0.8, 0.3, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "sparse",  # gemm type: dense or sparse
        "reference": 1 / 6,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 1,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "passed": True,
    },
    "21": {  # Fig.12.b.case9
        "setting": (2048, 1, 128, 0.8, 0.3, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "sparse",  # gemm type: dense or sparse
        "reference": 1 / 5,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 1,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "modeling mismatch": -0.466,  # manual recording
        "passed": False,
    },
    "22": {  # Fig.12.b.case10
        "setting": (256, 256, 2048, 0.8, 0.3, 128, 128, 128, 128, 128),
        # M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma (unit: per data)
        "type": "sparse",  # gemm type: dense or sparse
        "reference": 1 / 6,  # norm cycle ratio: sigma-to-tpu (reported in paper)
        "sigma_plan": 1,  # 0: (m-sta, n-str); 1: (m-str, n-sta); 2: min
        "passed": True,
    },
}

ratios = []
for case_id in case_ids:
    (M, N, K, sp_m, sp_n, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma) = cases[case_id]["setting"]
    sigma_plan = cases[case_id]["sigma_plan"]
    # for sp_m, sp_n in zip(sp_m_pool, sp_n_pool):
    assert sigma_plan in [0, 1, 2]  # input check
    print(
        f"configuration: (M, N, K, sparsity_M, sparsity_N, d1_tpu, d2_tpu, d1_sigma, d2_sigma, bw_sigma) = {cases[case_id]['setting']}")

    # calc TPU baseline performance (m-str, n-sta)
    parfor_n = min(d1_tpu, N)
    parfor_k = min(d2_tpu, K)
    ts_tpu = N * K / (parfor_n * parfor_k)  # temporal product of N * K
    loading = d2_tpu * ts_tpu  # loading clock cycles
    streaming = M * ts_tpu  # streaming clock cycles
    acc = (d2_tpu - 1) * ts_tpu  # extra accumulation cycles
    wb = ts_tpu  # cycles for writing back
    sum_tpu = loading + streaming + acc + wb  # total cycles

    # calc TPU baseline performance (m-sta, n-str)
    parfor_m = min(d1_tpu, M)
    parfor_k = min(d2_tpu, K)
    ts_tpu = M * K / (parfor_m * parfor_k)  # temporal product of M * K
    loading = d2_tpu * ts_tpu  # loading clock cycles
    streaming = N * ts_tpu  # streaming clock cycles
    acc = (d2_tpu - 1) * ts_tpu  # extra accumulation cycles
    wb = ts_tpu  # cycles for writing back
    sum_tpu2 = loading + streaming + acc + wb  # total cycles

    # SIGMA performance: (m-str, n-sta)
    N_nz = math.ceil(N * (1 - sp_n))
    M_nz = math.ceil(M * (1 - sp_m))
    if K < d2_sigma:  # calc parfor_k first
        parfor_k_sa = K
        parfor_n_sa = min(math.floor(d1_sigma * d2_sigma / parfor_k_sa), N_nz)
        ts_sa = math.ceil(N_nz * K / (parfor_n_sa * parfor_k_sa))  # temporal product of N * K
    else:  # calc paror_n first
        parfor_n_sa = min(d1_sigma, N_nz)
        # round parfor_n_sa to 8*x (assume that parfor must be 8*x)
        x = math.ceil(parfor_n_sa / 8)
        parfor_n_sa = 8 * (x)
        parfor_k_sa = min(math.floor(d1_sigma * d2_sigma / parfor_n_sa), K)
        if parfor_n_sa > N_nz:  # otherwise M_nz/parfor_m_sa2 will be <1, not realistic
            ts_sa = K / parfor_k_sa  # temporal product of N * K
        else:
            ts_sa = N_nz * K / (parfor_n_sa * parfor_k_sa)  # temporal product of N * K
    loading_sigma = math.ceil((parfor_n_sa * parfor_k_sa) / bw_sigma) * ts_sa
    streaming_sigma = M * ts_sa  # no skipping on streaming
    acc_sigma = math.log2(d2_sigma) * ts_sa
    wb_sigma = ts_sa
    sum_sigma = loading_sigma + streaming_sigma + acc_sigma + wb_sigma

    # SIGMA performance: (m-sta, n-str)
    N_nz = math.ceil(N * (1 - sp_n))
    M_nz = math.ceil(M * (1 - sp_m))
    if K < d2_sigma:  # calc parfor_k first
        parfor_k_sa2 = K
        parfor_m_sa2 = min(math.floor(d1_sigma * d2_sigma / parfor_k_sa2), M_nz)
        ts_sa2 = math.ceil(M_nz * K / (parfor_m_sa2 * parfor_k_sa2))  # temporal product of M * K
    else:
        parfor_m_sa2 = min(d1_sigma, M_nz)
        # round parfor_m_sa2 to 8*x (assume that parfor must be 8*x)
        x = math.ceil(parfor_m_sa2 / 8)
        parfor_m_sa2 = 8 * (x)
        parfor_k_sa2 = min(math.floor(d1_sigma * d2_sigma / parfor_m_sa2), K)
        if parfor_m_sa2 > M_nz:  # otherwise M_nz/parfor_m_sa2 will be <1, not realistic
            ts_sa2 = K / parfor_k_sa2  # temporal product of M * K
        else:
            ts_sa2 = M_nz * K / (parfor_m_sa2 * parfor_k_sa2)  # temporal product of N * K
    loading_sigma2 = math.ceil((parfor_m_sa2 * parfor_k_sa2) / bw_sigma) * ts_sa2
    streaming_sigma2 = N * ts_sa2
    acc_sigma2 = math.log2(d2_sigma) * ts_sa2
    wb_sigma2 = ts_sa2
    sum_sigma2 = loading_sigma2 + streaming_sigma2 + acc_sigma2 + wb_sigma2

    # derive the min
    sum_sigma_min = min(sum_sigma, sum_sigma2)
    if sum_sigma_min == sum_sigma:
        sum_tpu_min = sum_tpu
    else:
        sum_tpu_min = sum_tpu2
    # print("SIGMA: ", sum_sigma, sum_sigma2)
    # print("TPU: ", sum_tpu, sum_tpu2)

    # catch the results
    if sigma_plan == 0:
        sigma_cc = sum_sigma
        tpu_cc = sum_tpu
    elif sigma_plan == 1:
        sigma_cc = sum_sigma2
        tpu_cc = sum_tpu2
    else:
        sigma_cc = sum_sigma_min
        tpu_cc = sum_tpu_min

    modeling_ratio = sigma_cc / tpu_cc
    reference = cases[case_id]["reference"]

    # print out results
    print(f"Model: {modeling_ratio}, reference: {reference}", end=", ")
    print(f"Mismatch: {round((modeling_ratio / reference - 1), 3)}")
    if cases[case_id]["passed"]:
        ratios.append(round((modeling_ratio / reference), 3))

import matplotlib.pyplot as plt
import numpy as np

# Your data
x = [x for x in cases.keys() if (cases[x]["passed"] and int(x) >= 13)]
y = ratios
gemm = [f"{cases[i]['setting'][0]}-{cases[i]['setting'][1]}-{cases[i]['setting'][2]}" for i in x]
gemm.append("Average")
# calc aver y
mismatch_abs = 0
for v in y:
    mismatch_abs += abs(v-1)
mismatch_abs /= len(y)
y.append(1+mismatch_abs)
print(y)

# Create the bar chart
plt.figure(figsize=(5, 4))
bar_width = 0.4
x_plot = np.arange(len(gemm))
bars_sigma = plt.bar(x_plot-bar_width/2, np.ones(len(gemm)), color=u'#cbc0dd', width=bar_width, edgecolor='k', label='SIGMA')
bars_js = plt.bar(x_plot+bar_width/2, ratios, color=u'#fff6d5', width=bar_width, edgecolor='k', label='SunPar')
plt.xticks(range(len(gemm)), gemm)

# Customize the chart
# plt.title('Validation to SIGMA across Spase GeMMs (model/hardware)', fontsize=15, weight="bold")
plt.xlabel('GeMM Shape (M, N, K)', fontsize=14, weight="normal")
plt.ylabel('Norm Throughput', fontsize=14, weight="normal")

# Add value labels on top of each bar
for i in range(len(bars_js)):
    bar_js = bars_js[i]
    bar_sigma = bars_sigma[i]
    height = max(bar_js.get_height(), bar_sigma.get_height())
    plt.text(i, height,
            f'{round((ratios[i]-1)*100, 1)}%',
            ha='center', va='bottom', fontsize=10)

# Adjust y-axis to start slightly below the minimum value and end slightly above maximum
plt.ylim(0.8, 1.1)  # This gives some padding above and below the bars
# plt.grid(True, linestyle="-", alpha=0.7)
# plt.gca().set_axisbelow(True)
# plt.tick_params(axis='both', which='major', labelsize=12)  # set axis tick font size
plt.xticks(fontsize=12, rotation=45, ha='right')  # Rotates labels 45 degrees

plt.legend(loc='lower right', fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()