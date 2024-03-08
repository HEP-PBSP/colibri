import matplotlib.pyplot as plt
from reportengine.figure import figure, figuregen
import numpy as np
import pandas as pd
import glob
from colibri.plots_and_tables.plotting import get_fit_path
from scipy.stats import percentileofscore
import os

# Get the directory of the current script or module
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the style file relative to the current directory
style_file_path = os.path.join(current_directory, "ourstyle.mplstyle")

plt.style.use(style_file_path)


def gaussian_kl_divergence(x, y, symm=False):
    # Compute mean and covariance matrix
    mean_x = np.mean(x, axis=0)
    mean_y = np.mean(y, axis=0)

    cov_x = np.cov(x, rowvar=False)
    cov_y = np.cov(y, rowvar=False)

    if len(x.shape) == 1:
        mean_x = mean_x.reshape(1, 1)
        mean_y = mean_y.reshape(1, 1)
        cov_x = cov_x.reshape(1, 1)
        cov_y = cov_y.reshape(1, 1)

    # Compute the KL divergence
    kl_div = 0.5 * (
        np.log(np.linalg.det(cov_y) / np.linalg.det(cov_x))
        - len(mean_x)
        + np.trace(np.linalg.inv(cov_y) @ cov_x)
        + (mean_y - mean_x) @ np.linalg.inv(cov_y) @ (mean_y - mean_x)
    )

    # if symm is True, compute the symmetric KL divergence
    if symm:
        kl_div += 0.5 * (
            np.log(np.linalg.det(cov_x) / np.linalg.det(cov_y))
            - len(mean_x)
            + np.trace(np.linalg.inv(cov_x) @ cov_y)
            + (mean_x - mean_y) @ np.linalg.inv(cov_x) @ (mean_x - mean_y)
        )

    if len(x.shape) == 1:
        kl_div = kl_div[0, 0]

    return kl_div


def kl_div_test_resample(
    fit_A, fit_B, n_permutations=1000, symm=False, n_resample=100, fit_B_full=False
):
    """
    TODO
    """
    fit_A_path = get_fit_path(fit_A)
    fit_B_path = get_fit_path(fit_B)

    # Each folder has only one result file
    # Read the result file, no matter the type of fit
    df_A = pd.read_csv(glob.glob(fit_A_path + "/*_result.csv")[0], index_col=0)
    if fit_B_full:
        df_B = pd.read_csv(
            glob.glob(fit_B_path + "/ultranest_logs/chains/equal_weighted_post.txt")[0],
            sep=" ",
        )
    else:
        df_B = pd.read_csv(glob.glob(fit_B_path + "/*_result.csv")[0], index_col=0)

    x_A = df_A.values
    x_B = df_B.values

    kl_distro = []
    for i in range(n_permutations):
        x_A_aux = x_A[np.random.choice(x_A.shape[0], n_resample, replace=False)]
        x_B_aux = x_B[np.random.choice(x_B.shape[0], n_resample, replace=False)]
        kl_value = gaussian_kl_divergence(x_A_aux, x_B_aux, symm=symm)
        kl_distro.append(kl_value)
        # print(f"KL divergence between fit A and fit B: {kl_value}")

    kl_values_perm = []
    for i in range(n_permutations):
        perm_x, perm_y = permute_x_y_samples(
            x_A[:n_resample], x_B[:n_resample], random_seed=i
        )
        kl_values_perm.append(gaussian_kl_divergence(perm_x, perm_y, symm=symm))

    return {"kl_perm_distribution": kl_values_perm, "kl_distro": kl_distro}


def kl_div_test(fit_A, fit_B, n_permutations=1000, symm=False):
    """
    TODO
    """
    fit_A_path = get_fit_path(fit_A)
    fit_B_path = get_fit_path(fit_B)

    # Each folder has only one result file
    # Read the result file, no matter the type of fit
    df_A = pd.read_csv(glob.glob(fit_A_path + "/*_result.csv")[0], index_col=0)
    df_B = pd.read_csv(glob.glob(fit_B_path + "/*_result.csv")[0], index_col=0)

    x_A = df_A.values
    x_B = df_B.values

    kl_value = gaussian_kl_divergence(x_A, x_B, symm=symm)
    print(f"KL divergence between fit A and fit B: {kl_value}")

    kl_values_perm = []
    for i in range(n_permutations):
        perm_x, perm_y = permute_x_y_samples(x_A, x_B, random_seed=i)
        kl_values_perm.append(gaussian_kl_divergence(perm_x, perm_y, symm=symm))

    return {"kl_distribution": kl_values_perm, "kl_value": kl_value}


def kl_div_test_1D(fit_A, fit_B, n_permutations=1000):
    """
    TODO
    """
    fit_A_path = get_fit_path(fit_A)
    fit_B_path = get_fit_path(fit_B)

    # Each folder has only one result file
    # Read the result file, no matter the type of fit
    df_A = pd.read_csv(glob.glob(fit_A_path + "/*_result.csv")[0], index_col=0)
    df_B = pd.read_csv(glob.glob(fit_B_path + "/*_result.csv")[0], index_col=0)

    x_A = df_A.values
    x_B = df_B.values

    results = []

    for j in range(x_A.shape[1]):

        kl_value = gaussian_kl_divergence(x_A[:, j], x_B[:, j])

        kl_values_perm = []
        for i in range(n_permutations):
            perm_x, perm_y = permute_x_y_samples(x_A, x_B, random_seed=i)
            kl_values_perm.append(gaussian_kl_divergence(perm_x[:, j], perm_y[:, j]))

        results.append(
            {
                "kl_distribution": kl_values_perm,
                "kl_value": kl_value,
                "label": df_A.columns[j],
            }
        )

    return results


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
    perm_y = permute_x_y[x.shape[0] :, :]

    return perm_x, perm_y


@figure
def plot_kl_distribution_resample(kl_div_test_resample, n_resample=100):
    """
    TODO
    """
    kl_distribution = kl_div_test_resample["kl_perm_distribution"]
    kl_distro = kl_div_test_resample["kl_distro"]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.hist(
        kl_distribution,
        bins=20,
        color="blue",
        alpha=0.7,
        label="Perm. null distr.",
        density=True,
    )

    ax.hist(
        kl_distro,
        bins=20,
        color="red",
        alpha=0.7,
        density=True,
        label="Fits KL div.",
    )
    ax.set_xlabel("KL divergence", fontsize=18)
    ax.set_ylabel("Prob. density", fontsize=18)
    ax.set_title(
        "KL divergence distribution for resample size " + str(n_resample), fontsize=15
    )
    ax.legend(frameon=False, fontsize=15)

    return fig


@figure
def plot_kl_distribution(kl_div_test, title=""):
    """
    TODO
    """
    kl_distribution = kl_div_test["kl_distribution"]
    kl_value = kl_div_test["kl_value"]

    # Calculate the percentile rank
    percentile_rank = percentileofscore(kl_distribution, kl_value)

    # Calculate the p-value
    p_value = (100 - percentile_rank) / 100

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.hist(
        kl_distribution,
        bins=20,
        color="blue",
        alpha=0.7,
        label="Perm. null distr.",
        density=True,
    )
    ax.axvline(kl_value, color="red", linestyle="--", label="Fits KL div.")
    ax.set_xlabel("KL divergence", fontsize=18)
    ax.set_ylabel("Prob. density", fontsize=18)
    ax.set_title(title + f"\nPermutation test p-value: {p_value:.5f}", fontsize=15)
    ax.legend(frameon=False, fontsize=15)

    return fig


@figuregen
def plot_kl_distribution_1D(kl_div_test_1D):
    """
    TODO
    """

    for result in kl_div_test_1D:
        label = result["label"]

        yield plot_kl_distribution(result, title=f"KL divergence for component {label}")
