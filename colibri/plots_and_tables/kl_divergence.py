import matplotlib.pyplot as plt
from reportengine.figure import figure
import numpy as np
import pandas as pd
import os
from colibri.plots_and_tables.plotting import get_fit_path
from scipy.stats import percentileofscore


def gaussian_kl_divergence(x, y):
    # Compute mean and covariance matrix
    mean_x = np.mean(x, axis=0)
    mean_y = np.mean(y, axis=0)

    cov_x = np.cov(x, rowvar=False)
    cov_y = np.cov(y, rowvar=False)

    # Compute the KL divergence
    kl_div = 0.5 * (
        np.trace(np.linalg.inv(cov_y) @ cov_x)
        + (mean_y - mean_x) @ np.linalg.inv(cov_y) @ (mean_y - mean_x)
        - len(mean_x)
        + np.log(np.linalg.det(cov_y) / np.linalg.det(cov_x))
    )

    return kl_div


def kl_div_test(mc_fit, bayesian_fit, n_permutations=1000):
    """
    TODO
    """
    mc_path = get_fit_path(mc_fit)
    bayesian_path = get_fit_path(bayesian_fit)

    if os.path.exists(bayesian_path + "/ns_result.csv"):
        df_bayes = pd.read_csv(bayesian_path + "/ns_result.csv", index_col=0)

    if os.path.exists(mc_path + "/mc_result.csv"):
        df_mc = pd.read_csv(mc_path + "/mc_result.csv", index_col=0)

    x_bayes = df_bayes.values
    x_mc = df_mc.values

    kl_value = gaussian_kl_divergence(x_bayes, x_mc)
    print(f"KL divergence between MC and Bayesian fit: {kl_value}")

    kl_values_perm = []
    for i in range(n_permutations):
        perm_x, perm_y = permute_x_y_samples(x_mc, x_bayes, random_seed=i)
        kl_values_perm.append(gaussian_kl_divergence(perm_x, perm_y))

    return {"kl_distribution": kl_values_perm, "kl_value": kl_value}


def permute_x_y_samples(x, y, random_seed=0):
    """
    Permute samples of x with samples of y at random.
    Parameters
    ----------
    x: np.array
        First sample of vectors.
    y: np.array
        Second sample of vectors.
    """
    np.random.seed(random_seed)

    # check that the two samples are 2D arrays
    if len(x.shape) != 2:
        raise ValueError("The two samples must be 2D arrays.")

    concatenated_x_y = np.concatenate((x, y), axis=0)
    permute_x_y = np.random.permutation(concatenated_x_y)

    perm_x = permute_x_y[: x.shape[0], :]
    perm_y = permute_x_y[y.shape[0] :, :]

    return perm_x, perm_y


@figure
def plot_kl_distribution(kl_div_test):
    """
    TODO
    """
    kl_distribution = kl_div_test["kl_distribution"]
    kl_value = kl_div_test["kl_value"]

    # Calculate the percentile rank
    percentile_rank = percentileofscore(kl_distribution, kl_value)

    # Calculate the p-value
    p_value = (100 - percentile_rank) / 100

    fig, ax = plt.subplots()
    ax.hist(
        kl_distribution,
        bins=20,
        color="blue",
        alpha=0.7,
        label="Permutation distribution",
    )
    ax.axvline(kl_value, color="red", linestyle="--", label="KL divergence")
    ax.set_xlabel("KL divergence")
    ax.set_ylabel("Frequency")
    ax.set_title("KL divergence distribution")
    ax.set_title(f"Permutation test p-value: {p_value:.2f}")

    return fig
