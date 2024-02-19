import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from colibri.plots_and_tables.plotting import get_fit_path

from reportengine.figure import figure


def kl_divergence(x, y):
    """
    Computes the Kullback-Leibler divergence between two samples of vectors.
    # The kl divergence is computed as the average of the kl divergence between the
    # cartesian product of the two samples.

    Let x_i and y_i be two vectors of the same length, then the kl divergence
    between x_i and y_i is defined as: D_KL(x_i||y_i) = sum_i (x_i * log(x_i/y_i))

    The kl divergence between two samples of vectors is then defined as:

    D_KL(x||y) = sum_i (sum_j D_KL(x_i||y_j) / len(x)) / len(x)

    x and y are assumed to be 2D arrays of the same shape, where the first dimension
    is the length of the vectors and the second dimension is the number of vectors.

    Parameters
    ----------
    x: np.array
        First sample of vectors.

    y: np.array
        Second sample of vectors.
    """

    # check that the two samples have the same shape
    if x.shape != y.shape:
        raise ValueError("The two samples must have the same shape.")

    # check that the two samples are 2D arrays
    if len(x.shape) != 2:
        raise ValueError("The two samples must be 2D arrays.")

    # check that the two samples have the same length
    if x.shape[0] != y.shape[0]:
        raise ValueError("The two samples must have the same length.")

    # compute the kl divergence
    kl = np.mean([[np.sum(x_i * np.log(x_i / y)) for x_i in x.T] for y in y.T])

    return kl


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

    # check that the two samples have the same shape
    if x.shape != y.shape:
        raise ValueError("The two samples must have the same shape.")

    # check that the two samples are 2D arrays
    if len(x.shape) != 2:
        raise ValueError("The two samples must be 2D arrays.")

    # check that the two samples have the same length
    if x.shape[0] != y.shape[0]:
        raise ValueError("The two samples must have the same length.")

    concatenated_x_y = np.concatenate((x, y), axis=1)
    permute_idx = np.random.permutation(concatenated_x_y.shape[1])
    permute_x_y = concatenated_x_y[:, permute_idx]

    perm_x = permute_x_y[:, : x.shape[1]]
    perm_y = permute_x_y[:, x.shape[1] :]

    return perm_x, perm_y


def kl_div_test(mc_fit, bayesian_fit, n_permutations=500):
    """
    TODO
    """
    mc_path = get_fit_path(mc_fit)
    bayesian_path = get_fit_path(bayesian_fit)

    if os.path.exists(bayesian_path + "/ns_result.csv"):
        df_bayes = pd.read_csv(bayesian_path + "/ns_result.csv", index_col=0)

    if os.path.exists(mc_path + "/mc_result.csv"):
        df_mc = pd.read_csv(mc_path + "/mc_result.csv", index_col=0)

    # parton distribution functions are not really PDFs, hence they can be negative
    # we take the absolute value of the samples to compute the KL divergence
    # this is an approximation, is it a good one?
    x_bayes = np.abs(df_bayes.values.T)
    x_mc = np.abs(df_mc.values.T)
    
    kl_value = kl_divergence(x_mc, x_bayes)
    print(f"KL divergence between MC and Bayesian fit: {kl_value}")

    kl_values_perm = []
    for i in range(n_permutations):
        perm_x, perm_y = permute_x_y_samples(x_mc, x_bayes, random_seed=i)
        kl_values_perm.append(kl_divergence(perm_x, perm_y))

    return {"kl_distribution": np.array(kl_values_perm), "kl_value": kl_value}


@figure
def plot_kl_distribution(kl_div_test):
    """
    TODO
    """
    kl_distribution = kl_div_test["kl_distribution"]
    kl_value = kl_div_test["kl_value"]

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

    return fig
