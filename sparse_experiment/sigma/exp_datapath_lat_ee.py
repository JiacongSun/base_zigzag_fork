import math
import numpy as np
import logging
import matplotlib.pyplot as plt
import scipy.stats as stats
from sigma_like import exp_sigma


def plot_datapath(config_collect, pe_count_collect, datapath_util_collect_mu,
                  datapath_util_collect_std,
                  lat_datapath_collect_mu,
                  lat_datapath_collect_std,
                  ee_datapath_collect_mu,
                  ee_datapath_collect_std):
    # Data organized by configurations
    configs = {
        ('gating', 'gating'): {
            'pe_count': pe_count_collect,
            'util_mu': [],
            'util_std': [],
            'lat_mu': [],
            'lat_std': [],
            'ee_mu': [],
            'ee_std': []
        },
        ('gating', 'skipping'): {
            'pe_count': pe_count_collect,
            'util_mu': [],
            'util_std': [],
            'lat_mu': [],
            'lat_std': [],
            'ee_mu': [],
            'ee_std': []
        },
        ('skipping', 'gating'): {
            'pe_count': pe_count_collect,
            'util_mu': [],
            'util_std': [],
            'lat_mu': [],
            'lat_std': [],
            'ee_mu': [],
            'ee_std': []
        },
        ('skipping', 'skipping'): {
            'pe_count': pe_count_collect,
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
        configs[config]['util_mu'] = datapath_util_collect_mu[
                                     sample_count * saf_config_index:sample_count * (saf_config_index + 1)]
        configs[config]['util_std'] = datapath_util_collect_std[
                                      sample_count * saf_config_index:sample_count * (saf_config_index + 1)]
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))

    # Colors and markers for different configurations
    styles = {
        ('gating', 'gating'): {'color': 'blue', 'marker': 'o', 'label': 'gating-gating'},
        ('gating', 'skipping'): {'color': 'red', 'marker': 's', 'label': 'gating-skipping'},
        ('skipping', 'gating'): {'color': 'green', 'marker': '^', 'label': 'skipping-gating'},
        ('skipping', 'skipping'): {'color': 'purple', 'marker': 'D', 'label': 'skipping-skipping'}
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
    ax1.set_ylabel('Latency [cc]', fontsize=10)

    for saf_config_index in range(len(config_collect)):
        config = config_collect[saf_config_index]
        style = styles[config]
        ax1.errorbar(configs[config]['pe_count'], configs[config]['lat_mu'],
                     yerr=configs[config]['lat_std'],
                     color=style['color'], marker=style['marker'],
                     label=style['label'], markersize=5,
                     **error_bar_props)

    ax1.grid(True, alpha=0.8)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_yscale('log')

    # Plot 2: Energy vs PE Count
    # ax2.set_title('Energy vs PE Count', pad=15, fontsize=12)
    ax2.set_xlabel('PE Count', fontsize=10)
    ax2.set_ylabel('Energy [pJ]', fontsize=10)

    for saf_config_index in range(len(config_collect)):
        config = config_collect[saf_config_index]
        style = styles[config]
        ax2.errorbar(configs[config]['pe_count'], configs[config]['ee_mu'],
                     yerr=configs[config]['ee_std'],
                     color=style['color'], marker=style['marker'],
                     label=style['label'], markersize=5,
                     **error_bar_props)

    ax2.grid(True, alpha=0.8)
    # ax2.legend(fontsize=10, loc='upper right')

    # Plot 3: Utilization vs PE Count
    # ax3.set_title('Utilization vs PE Count', pad=15, fontsize=12)
    # ax3.set_xlabel('PE Count', fontsize=10)
    # ax3.set_ylabel('Utilization', fontsize=10)
    #
    # for saf_config_index in range(len(config_collect)):
    #     config = config_collect[saf_config_index]
    #     style = styles[config]
    #     ax3.errorbar(configs[config]['pe_count'], configs[config]['util_mu'],
    #                  yerr=configs[config]['util_std'],
    #                  color=style['color'], marker=style['marker'],
    #                  label=style['label'], markersize=5,
    #                  **error_bar_props)
    #
    # ax3.grid(True, alpha=0.8)
    # ax3.legend(fontsize=10, loc='lower left')

    # Set x-axis to log scale for all plots since PE count varies exponentially
    ax1.set_xscale('log', base=2)
    ax2.set_xscale('log', base=2)
    # ax3.set_xscale('log', base=2)

    # Adjust layout and spacing
    plt.tight_layout()
    plt.show()


def calc_mean_var_std_for_truncated_normal_distribution(mu, std, lower_clip, upper_clip):
    # formula reference: https://en.wikipedia.org/wiki/Truncated_normal_distribution (section: Properties->Moments)
    # lower_clip/upper_clip == None: means negtive/postive infinite
    if std == 0:
        return mu, 0, 0
    if lower_clip is None:
        if upper_clip is None:
            var = std ** 2
            return mu, var, std
        else:  # upper clip
            beta = (upper_clip - mu) / std
            beta_pdf = stats.norm.pdf(beta)
            beta_cdf = stats.norm.cdf(beta)
            new_mean = mu - std * (beta_pdf / beta_cdf)
            new_var = (std ** 2) * (1 - beta * beta_pdf / beta_cdf - (beta_pdf / beta_cdf) ** 2)
            new_std = new_var ** 0.5
            return new_mean, new_var, new_std
    else:
        if upper_clip is None:  # lower clip
            alpha = (lower_clip - mu) / std
            alpha_pdf = stats.norm.pdf(alpha)
            alpha_cdf = stats.norm.cdf(alpha)
            Z = 1 - alpha_cdf
            new_mean = mu + std * alpha_pdf / Z
            new_var = (std ** 2) * (1 + alpha * alpha_pdf / Z - (alpha_pdf / Z) ** 2)
            new_std = new_var ** 0.5
            return new_mean, new_var, new_std
        else:  # two sides clip
            alpha = (lower_clip - mu) / std
            beta = (upper_clip - mu) / std
            alpha_pdf = stats.norm.pdf(alpha)
            alpha_cdf = stats.norm.cdf(alpha)
            beta_pdf = stats.norm.pdf(beta)
            beta_cdf = stats.norm.cdf(beta)
            new_mean = mu - std * ((beta_pdf - alpha_pdf) / (beta_cdf - alpha_cdf))
            new_var = (std ** 2) * (1 - ((beta * beta_pdf - alpha * alpha_pdf) / (beta_cdf - alpha_cdf)) - (
                    (beta_pdf - alpha_pdf) / (beta_cdf - alpha_cdf)) ** 2)
            new_std = new_var ** 0.5
            return new_mean, new_var, new_std


def calc_mean_var_std_for_max_norm_with_constant(mu, std, constant):
    # This function is to calc the mean, var, std for: max(constant, N)
    # where N is a norm distribution with mean=mu, sigma=std
    # step 1: calc the prob of N < constant (prob_a)
    assert std >= 0
    if std == 0:
        prob_a = 1 if mu <= constant else 0
    else:
        prob_a = stats.norm.cdf((constant - mu) / std)
        prob_a = round(prob_a, 3)
    if prob_a == 0:
        new_mu = mu
        new_std = std
        new_var = std ** 2
        return new_mu, new_var, new_std
    elif prob_a == 1:
        mean_a = constant
        var_a = 0
        std_a = 0
        return mean_a, var_a, std_a
    else:
        prob_b = 1 - prob_a
        # step 2: calc the mean, var, std for a constant distribution
        mean_a = constant
        var_a = 0
        std_a = 0
        # step 3: calc the mean, var, std for truncated N distribution
        mean_b, var_b, std_b = calc_mean_var_std_for_truncated_normal_distribution(
            mu=mu,
            std=std,
            lower_clip=constant,
            upper_clip=None
        )
        # step 4: calc the mean, var, std for mixture distribution
        # formula reference: https://en.wikipedia.org/wiki/Mixture_distribution#Finite_and_countable_mixtures
        # reference section: Properties -> Moments (the case of a mixture of one-dimensional distribution)
        new_mean = prob_a * mean_a + prob_b * mean_b
        new_var = prob_a * (var_a + mean_a ** 2) + prob_b * (var_b + mean_b ** 2) - new_mean ** 2
        new_std = new_var ** 0.5
        return new_mean, new_var, new_std


def calc_mean_var_std_for_min_norm_with_constant(mu, std, constant):
    # This function is to calc the mean, var, std for: min(constant, N)
    # where N is a norm distribution with mean=mu, sigma=std
    # step 1: calc the prob of N < constant (prob_a)
    assert std >= 0
    if std == 0:
        prob_a = 1 if mu <= constant else 0
    else:
        prob_a = stats.norm.cdf((constant - mu) / std)
        prob_a = round(prob_a, 3)
    prob_b = 1 - prob_a
    if prob_b == 0:
        new_mu = mu
        new_std = std
        new_var = std ** 2
        return new_mu, new_var, new_std
    elif prob_b == 1:
        mean_a = constant
        var_a = 0
        std_a = 0
        return mean_a, var_a, std_a
    else:
        # step 2: calc the mean, var, std for a constant distribution
        mean_a = constant
        var_a = 0
        std_a = 0
        # step 3: calc the mean, var, std for truncated N distribution
        mean_b, var_b, std_b = calc_mean_var_std_for_truncated_normal_distribution(
            mu=mu,
            std=std,
            lower_clip=None,
            upper_clip=constant
        )
        # step 4: calc the mean, var, std for mixture distribution
        # formula reference: https://en.wikipedia.org/wiki/Mixture_distribution#Finite_and_countable_mixtures
        # reference section: Properties -> Moments (the case of a mixture of one-dimensional distribution)
        new_mean = prob_b * mean_a + prob_a * mean_b
        new_var = prob_b * (var_a + mean_a ** 2) + prob_a * (var_b + mean_b ** 2) - new_mean ** 2
        new_std = new_var ** 0.5
        return new_mean, new_var, new_std


def calc_mean_std_of_mult_distribution(mu1, mu2, std1, std2, cov=0):
    """ calc the mean and std of z=norm(mu1, std1) * norm(mu2, std2) """
    """ cov: covariance between the two distribution """
    mu = mu1 * mu2 + cov
    var = (std1 * std2) ** 2 + (std1 * mu2) ** 2 + (std2 * mu1) ** 2
    std = var ** 0.5
    return mu, std


if __name__ == "__main__":
    """
    Exp: mem utilization (512 KB) @ L2, ResNet18
    """
    logging_level = logging.INFO  # logging level
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)
    # resnet18, l2
    ox = 56
    oy = 56
    c = 64
    fx = 3
    fy = 3
    k = 64
    average_density = 0.59813  # for activation
    density_std = 0.08958  # for activation
    weight_density = 0.9  # sparse on C dim (from man's network analysis)
    # vgg19, l3
    ox = 56
    oy = 56
    c = 256
    fx = 3
    fy = 3
    k = 256
    average_density = 0.4539205702647658  # for activation
    density_std = 0.051416986705508046  # for activation
    weight_density = 0.8  # sparse on C dim (from https://sparsezoo.neuralmagic.com/models/vgg-19-imagenet-pruned?hardware=deepsparse-c6i.12xlarge&comparison=vgg-19-imagenet-base)
    weight_density_std = 0

    mac_ee_unit = 0.01205346455 * 7 * 2  # extracted from bitwave, pJ/mac
    mac_ee_skip_control = 0.03831114971 * 2  # extracted from bitwave, pJ/mac
    # for lat/ee calc
    saf_pool = [{"I": "gating", "W": "gating"}, {"I": "gating", "W": "skipping"},
                {"I": "skipping", "W": "gating"}, {"I": "skipping", "W": "skipping"}]
    arch_d1_pool = [4, 8, 16, 32]
    arch_d2_pool = [4, 8, 16, 32]

    config_collect = [(x["I"], x["W"]) for x in saf_pool]
    pe_count_collect = []
    datapath_util_collect_mu = []
    datapath_util_collect_std = []
    lat_datapath_collect_mu = []
    lat_datapath_collect_std = []
    ee_datapath_collect_mu = []
    ee_datapath_collect_std = []

    dense_mac_count = ox * oy * c * fx * fy * k

    # calc std when both i and w has non-zero std (such as for transformers)
    density_total_mu, density_total_std = calc_mean_std_of_mult_distribution(mu1=average_density,
                                                                             mu2=weight_density,
                                                                             std1=density_std,
                                                                             std2=weight_density_std,
                                                                             cov=0)

    for saf in saf_pool:
        saf_i = saf["I"]
        saf_w = saf["W"]
        for arch_size_d1, arch_size_d2 in zip(arch_d1_pool, arch_d2_pool):
            pe_count = arch_size_d1 * arch_size_d2
            # calc sparse_mac_count
            if saf_i == "gating" and saf_w == "gating":
                sparse_mac_count = dense_mac_count
                sparse_mac_count_std = 0
            elif saf_i == "gating" and saf_w == "skipping":
                sparse_mac_count = dense_mac_count * weight_density
                sparse_mac_count_std = dense_mac_count * weight_density_std
            elif saf_i == "skipping" and saf_w == "gating":
                sparse_mac_count = dense_mac_count * average_density
                sparse_mac_count_std = dense_mac_count * density_std
            else:
                sparse_mac_count = dense_mac_count * density_total_mu
                sparse_mac_count_std = dense_mac_count * density_total_std
            # calc spatial_unrolling_size: mu and std (weight stationary)
            if saf_i == "gating" and saf_w == "gating":
                spatial_unrolling_d1 = min(arch_size_d1, k)
                spatial_unrolling_d2 = min(arch_size_d2, c)
                spatial_unrolling_size = spatial_unrolling_d1 * spatial_unrolling_d2
                spatial_unrolling_size_std = 0
            else:  # "skipping" in [saf_i, saf_w]
                spatial_unrolling_size, __, spatial_unrolling_size_std = calc_mean_var_std_for_min_norm_with_constant(
                    mu=sparse_mac_count,
                    std=sparse_mac_count_std,
                    constant=pe_count
                )
            # calc datapath util
            datapath_util_mu = spatial_unrolling_size / pe_count
            datapath_util_std = spatial_unrolling_size_std / pe_count
            # calc latency (cycle count): mu and std
            if saf_i == "gating" and saf_w == "gating":
                lat_datapath = sparse_mac_count / spatial_unrolling_size
                lat_datapath_std = 0
            else:
                lat_datapath_before_ceiling = sparse_mac_count / pe_count
                lat_datapath_std_before_ceiling = sparse_mac_count_std / pe_count
                lat_datapath, lat_datapath_std = exp_sigma.ceil_distribution_stats(
                    mu=lat_datapath_before_ceiling,
                    sigma=lat_datapath_std_before_ceiling)
            # calc energy: mu and std
            true_sparse_mac_count = dense_mac_count * density_total_mu
            true_sparse_mac_count_std = dense_mac_count * density_total_std
            if saf_i == "gating" and saf_w == "gating":
                ee_skip_overhead_factor = 0
            elif (saf_i == "gating" and saf_w == "skipping") or (saf_i == "skipping" and saf_w == "gating"):
                # single-side skipping
                ee_skip_overhead_factor = 1
            else:
                # dual-side skipping
                ee_skip_overhead_factor = 2
            ee_datapath = (mac_ee_unit + ee_skip_overhead_factor * mac_ee_skip_control) * true_sparse_mac_count
            ee_datapath_std = (mac_ee_unit + ee_skip_overhead_factor * mac_ee_skip_control) * true_sparse_mac_count_std
            # collect data
            pe_count_collect.append(pe_count)
            datapath_util_collect_mu.append(datapath_util_mu)
            datapath_util_collect_std.append(datapath_util_std)
            lat_datapath_collect_mu.append(lat_datapath)
            lat_datapath_collect_std.append(lat_datapath_std)
            ee_datapath_collect_mu.append(ee_datapath)
            ee_datapath_collect_std.append(ee_datapath_std)
            logging.info(f"saf_i: {saf_i}, saf_w: {saf_w}, pe_count: {pe_count}, util_mu: {datapath_util_mu}, "
                         f"util_std: {datapath_util_std}, lat_mu: {lat_datapath}, lat_std: {lat_datapath_std}, "
                         f"3lat_std/lat_mu: {3 * lat_datapath_std / lat_datapath}, "
                         f"ee_mu: {ee_datapath}, ee_std: {ee_datapath_std}, "
                         f"3ee_std/ee_mu: {3 * ee_datapath_std / ee_datapath}")
    pe_count_collect = pe_count_collect[:len(set(pe_count_collect))]
    plot_datapath(config_collect, pe_count_collect, datapath_util_collect_mu,
                  datapath_util_collect_std,
                  lat_datapath_collect_mu,
                  lat_datapath_collect_std,
                  ee_datapath_collect_mu,
                  ee_datapath_collect_std)
