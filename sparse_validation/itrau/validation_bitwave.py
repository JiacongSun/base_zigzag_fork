# validation to BitWave
OXu = 4
Ku = 32
Cu = 32
op_precision = 8
freq = 71.4  # MHz

# interpolated value
pj_per_op = 0.01205346455 * 7  # pJ, interpolation in least-squares method
pj_ctrl = 0.03831114971  # pJ

ee = []
tp = []

for zl in range(0, op_precision - 1):
    # ss = 2
    # ideal = 1 * (7 - zl)
    # tm_util = ideal / (ideal + ss)

    # calc tm utilization
    (ox, oy, c, k, fx, fy) = (28, 28, 128, 128, 3, 3)
    # interpolate ss
    ss_dict = {  # zl: ss
        "0": 2,
        "1": 3,
        "2": 4,
        "3": 5,
        "4": 6,
        "5": 6,
        "6": 7,
    }
    assert str(zl) in ss_dict.keys()
    ss = ss_dict[str(zl)]
    cc_stall = ss * fy * c * k * ox * oy / (512 * 8)
    ideal_cc = ox * oy * c * k * fx * fy * (7 - zl) / (512 * 8)
    tm_util = ideal_cc / (ideal_cc + cc_stall)
    print(f"temporal utilization: {round(tm_util, 2)}", end=", ")

    macs_per_cycle = OXu * Ku * Cu / (7 - zl)  # assume 100% PE utilization and no memory stall
    macs_per_cycle = macs_per_cycle * tm_util  # penalty term due to tm utilization
    op_per_cycle = 2 * macs_per_cycle
    # calc throughput
    bops_per_cycle = op_per_cycle * op_precision * op_precision
    btops = bops_per_cycle / (1e6 / freq)
    # calc energy
    topsw = 1 / (pj_per_op / 7 * (7 - zl) + pj_ctrl)
    btopsw = topsw * op_precision * op_precision
    ee.append(round(btopsw, 2))
    tp.append(round(btops, 2))
    print(f"ZL: {zl}, BTOPs: {round(btops, 2)}, BTOPS/W: {round(btopsw, 2)}")

import matplotlib.pyplot as plt
import numpy as np

# Data
x = np.arange(7)
bars = [538.38, 573.73, 639.35, 719.13, 816.07, 1079.76, 1320.84]  # BTOPs/W
line = [4.86, 5.18, 5.73, 6.41, 7.29, 9.58, 11.66]  # BTOPs
mm_ee = (np.array(ee) / np.array(bars) - 1) * 100  # %
mm_tp = (np.array(tp) / np.array(line) - 1) * 100  # %
mm_ee_abs = abs((np.array(ee) / np.array(bars) - 1) * 100)  # %
mm_tp_abs = abs((np.array(tp) / np.array(line) - 1) * 100)  # %
print(f"The average throughput mismatch: {round(sum(mm_tp_abs)/len(mm_tp_abs), 2)}%")
print(f"The average energy mismatch: {round(sum(mm_ee_abs)/len(mm_ee_abs), 2)}%")

# print(mm_tp, f"{np.average(mm_tp)}%")

# Create a figure with two y-axes
fig, ax1 = plt.subplots(figsize=(5, 3))
ax2 = ax1.twinx()

# Plot bars
offset = 0.2
bars_plot_ms = ax1.bar(x/8-offset/8, bars, color=u'#eec458', width=0.05, edgecolor='k', label='BitWave')
bars_plot_js = ax1.bar(x/8+offset/8, ee, color='#fff6d5', width=0.05, edgecolor='k', label='SunPar')

ax1.set_yscale('log')
ax1.set_ylim(1, max(max(ee), max(bars))*2)
ax1.set_ylabel('Energy Efficiency [BTOPS/W]', fontsize=12, weight='normal')
ax1.set_xlabel('Sparsity Level', fontsize=14, weight='normal')

# Plot line
line_plot_ms = ax2.plot(x/8, line, color=u'#000000', marker='o', linewidth=2, markersize=6, markeredgecolor='white', label='BitWave')
line_plot_js = ax2.plot(x/8, tp, '--', color=u'#b32828', marker='s', linewidth=2, markersize=6, markeredgecolor='white', label='SunPar')

ax2.set_ylabel('Throughput [BTOPS]', color=u'#b32828', fontsize=12, weight='normal')
ax2.tick_params(axis='y', labelcolor=u'#b32828')
ax2.set_ylim(0, max(max(tp), max(line))*1.1)

# Add bar value annotations
for i, v in enumerate(bars):
    ax1.text(i/8, v*1.05, f'{str(round(mm_ee[i], 1))}%', ha='center', va='bottom', fontsize=10, weight='normal', rotation=0)
# for i, v in enumerate(ee):
#     mm = round(mm_ee[i], 0)
    # ax1.text(i, v*1.1, f'{str(v)}', ha='center', va='bottom', fontsize=12, weight='bold', rotation=0)
    # ax1.text(i, v*1.2, f'{str(round(mm_ee[i], 1))}%', ha='center', va='bottom', fontsize=12, weight='bold', rotation=0)

# Add line value annotations
for i, v in enumerate(tp):
    if i < 5:
        y_offset = 0.8
    else:
        y_offset = 0.85
    ax2.text(i/8, v*y_offset, f'{str(round(mm_tp[i], 1))}%', ha='center', color=u'#b32828', fontsize=10, weight='normal', rotation=0,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# Customize x-axis
# ax2.set_xlabel('Sparsity level', fontsize=14, weight='normal')
ax2.set_xticks(x/8)

# Add title
# plt.title('Chip Characteristic #Sparse Line', weight='bold', fontsize=15)

# Add grid
# ax2.grid(True, alpha=0.3, c='w', which='both')
# ax2.set_axisbelow(True)

# Adjust background color
ax2.set_facecolor(u'#f0f0f0')  # Set plot area background color

# Add legend
plots = (bars_plot_ms, bars_plot_js, line_plot_ms[0], line_plot_js[0])
labels = [plot.get_label() for plot in plots]
ax1.legend(plots, labels, loc='lower right', ncol=2, fontsize=12)

# Adjust layout
plt.tight_layout()

plt.show()