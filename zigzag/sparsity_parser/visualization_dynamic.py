import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from inference_hooker import NetworkInference
import numpy as np
from torchvision.io import read_image

############################################
# Global parameter setting
tile_size = 8  # targeted tile size
layer_idx = 2  # targeted layer
img_numbers = 100  # sample counts
model_name = "resnet18"  # targeted network, options: [resnet18, resnet50, vgg19, mobilenetv2, mobilenetv3, quant_mobilenetv2]
dataset_name = "imagenet"  # targeted dataset, options: [cifar10, imagenet]
operand = "i"  # i or w
plot_all_layers = False  # whether or not plot all layers at the same time
############################################

# Set up counter
counter = 0
legal_img_counter = 0
density_mean_mean = 0
density_std_mean = 0

# Set up the figure, the axis, and the plot elements we want to animate
fig, ax = plt.subplots(ncols=2)
ax[0].set_xlim(0, 1.1)
ax[0].set_ylim(0, 1.1)
ax[0].set_xlabel("Tile density ($ll_{density}$)")
ax[0].set_ylabel("Tile occurrence ($P_{density}$)")
ax[0].set_title(f"Tile density distribution\n(tile: {tile_size}, layer: {layer_idx})")
ax[0].grid(True)
ax[0].set_axisbelow(True)
line, = ax[0].plot([], [], "--o", color='black', markerfacecolor="moccasin",
                   markeredgecolor='black')
mean_line, = ax[0].plot([], [], "--^", color='black', markerfacecolor="black",
                        markeredgecolor='black')

ax[1].set_xlim(0.0, 1.1)
ax[1].set_ylim(0.0, 0.6)
ax[1].set_xlabel("Tile density mean $\mu ll_{density}$")
ax[1].set_ylabel("Tile density std $\sigma ll_{density}$")
ax[1].set_title(f"Tile density std vs. mean\n(img number: {img_numbers})")
line_mean2, = ax[1].plot([], [], "--^", color='black', markerfacecolor="black",
                         markeredgecolor='black', linewidth=1)
line_std2, = ax[1].plot([], [], "-->", color='black', markerfacecolor="black",
                        markeredgecolor='black', linewidth=1)
ax[1].grid(True)
ax[1].set_axisbelow(True)
plt.tight_layout()

# Set up color list
colors = plt.cm.get_cmap("tab20", 25)
color_list = [colors(i) for i in range(25)]

# Set up legend
# legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_list[i], markersize=8, label=f'layer{i}') for i in range(1, 21)]
# ax[0].legend(handles=legend_elements, loc='best', ncol=2)

# Set up network and image list
nn = NetworkInference(model_name=model_name, dataset_name=dataset_name)
if dataset_name == "cifar10":
    img_indices = np.random.randint(0, 10000, size=img_numbers)
else:  # imagenet
    img_indices = np.random.randint(1, 40000, size=img_numbers)


def init():
    line.set_data([], [])
    mean_line.set_data([], [])
    return line, mean_line,


def update_i(frame):
    global counter
    global legal_img_counter
    global density_mean_mean
    global density_std_mean
    global layer_idx
    global img_numbers
    global plot_all_layers
    global dataset_name
    counter += 1
    if dataset_name == "cifar10":
        img_name = None
        legal_img_counter += 1
    else:
        img_name = NetworkInference.convert_imagenet_idx_to_filename(img_idx=frame)
        if read_image(img_name).shape[0] != 3:
            print("*****")
            line.set_data([], [])
            mean_line.set_data([], [])
            return line, mean_line,
        else:
            legal_img_counter += 1

    if plot_all_layers:
        print(f"counter: {counter:<4}, img: {frame:<5}, layer: {layer_idx}", end=",\n")
        # return None to unused figures
        line.set_data([], [])
        mean_line.set_data([], [])
        line_mean2.set_data([], [])
        line_std2.set_data([], [])
        # fetch results for every layer
        intermediate_acts: list = nn.extract_activation_of_an_intermediate_layer(layer_idx=layer_idx,
                                                                                 img_idx=frame,
                                                                                 img_name=img_name,
                                                                                 return_all_layers=True)
        # define color maps
        colors_all_layers = np.random.rand(len(intermediate_acts))
        # extract mean and std per layer for a single image
        mean_points = []
        std_points = []
        for intermediate_act in intermediate_acts:
            density_list, density_occurrence, density_mean, density_std = NetworkInference.calc_density_distribution(
                op_array=intermediate_act,
                tile_size=tile_size,
                enable_relu=True)
            mean_points.append(density_mean)
            std_points.append(density_std)
        # plot scatter
        ax[1].scatter(mean_points, std_points, s=12, c=colors_all_layers, cmap="viridis", alpha=0.7, edgecolors="black")
    else:
        intermediate_act = nn.extract_activation_of_an_intermediate_layer(layer_idx=layer_idx,
                                                                          img_idx=frame,
                                                                          img_name=img_name)
        density_list, density_occurrence, density_mean, density_std = NetworkInference.calc_density_distribution(
            op_array=intermediate_act,
            tile_size=tile_size,
            enable_relu=True)
        line.set_data(density_list, density_occurrence)
        mean_line.set_data([density_mean, density_mean], [0, 2])
        ax[1].scatter(density_mean, density_std, s=12, c=color_list[layer_idx % len(color_list)], edgecolors="black")

        density_mean_mean = (density_mean_mean * (legal_img_counter - 1) + density_mean) / legal_img_counter
        density_std_mean = ((density_std_mean * (legal_img_counter - 1)) + density_std) / legal_img_counter
        line_mean2.set_data([density_mean_mean, density_mean_mean], [0, 2])
        line_std2.set_data([0, 2], [density_std_mean, density_std_mean])
        print(f"counter: {counter:<4}, img: {frame:<5}, layer: {layer_idx}, density mean: {density_mean:<4}, std: {density_std:<4}", end="\n")

    return line, mean_line, ax[1], line_mean2, line_std2,


def update_w(frame):
    global plot_all_layers
    if plot_all_layers:
        # return None to unused figures
        line.set_data([], [])
        mean_line.set_data([], [])
        line_mean2.set_data([], [])
        line_std2.set_data([], [])
        # fetch results for every layer
        weights = nn.extract_weight_of_an_intermediate_layer(layer_idx=1,
                                                             return_all_layers=True)
        # define color maps
        colors_all_layers = np.random.rand(len(weights))
        # extract mean and std per layer for a single image
        mean_points = []
        std_points = []
        for weight in weights:
            density_list, density_occurrence, density_mean, density_std = NetworkInference.calc_density_distribution(
                op_array=weight,
                tile_size=tile_size,
                enable_relu=False)
            mean_points.append(density_mean)
            std_points.append(density_std)
        # plot scatter
        ax[1].scatter(mean_points, std_points, s=12, c=colors_all_layers, cmap="viridis", alpha=0.7,
                      edgecolors="black")
    else:
        line_mean2.set_data([], [])
        line_std2.set_data([], [])
        # fetch results for targeted layer
        weight = nn.extract_weight_of_an_intermediate_layer(layer_idx=layer_idx,
                                                            return_all_layers=False)
        density_list, density_occurrence, density_mean, density_std = NetworkInference.calc_density_distribution(
            op_array=weight,
            tile_size=tile_size,
            enable_relu=False)
        line.set_data(density_list, density_occurrence)
        mean_line.set_data([density_mean, density_mean], [0, 2])
        ax[1].scatter(density_mean, density_std, s=12, c=color_list[layer_idx % len(color_list)], edgecolors="black")
    return line, mean_line, ax[1], line_mean2, line_std2,


# Call the animator
# :param blit: for fast rendering
# :param interval: delay per animation (ms)
if operand == "i":
    anim = FuncAnimation(fig, update_i, frames=img_indices, init_func=init, interval=100, blit=True, repeat=False)
else:
    img_indices = np.random.randint(1, 2, size=1)  # not used
    anim = FuncAnimation(fig, update_w, frames=img_indices, init_func=init, interval=100, blit=True, repeat=False)
plt.show()

# To save the animation to a file
# anim.save(f'animation_layer{layer_idx}_tile_{tile_size}_samples_{img_numbers}.mp4', writer='ffmpeg')

# To display the animation in a Jupyter notebook
# from IPython.display import HTML
# HTML(anim.to_jshtml())
