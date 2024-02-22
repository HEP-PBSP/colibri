import os
from scipy.stats import permutation_test
import numpy as np
import matplotlib.pyplot as ptl
import pandas as pd

from reportengine.figure import figure, figuregen
from colibri.plots_and_tables.plotting import get_fit_path


@figuregen
def plot_perm_test(mc_fit, bayesian_fit, perm_settings):
    """
    Plot the permutation test for two independent samples.

    Parameters
    ----------
    x: np.array
        First sample of vectors.

    y: np.array
        Second sample of vectors.

    kwargs: dict
        Keyword arguments for the permutation_test function from scipy.stats.
    """

    mc_path = get_fit_path(mc_fit)
    bayesian_path = get_fit_path(bayesian_fit)

    if not "statistic" in perm_settings:
        raise KeyError(
            "The 'statistic' key must be present in the perm_settings dictionary."
        )

    if perm_settings["statistic"] == "mean":
        perm_settings["statistic"] = lambda x, y, axis: np.mean(np.mean(x, axis=axis) - np.mean(
            y, axis=axis
        ))
        name_statistic = "mean"
    elif perm_settings["statistic"] == "var":
        perm_settings["statistic"] = lambda x, y, axis: np.mean(np.var(x, axis=axis) - np.var(
            y, axis=axis
        ))
        name_statistic = "variance"
    elif perm_settings["statistic"] == "euclid":
        perm_settings["statistic"] = lambda x, y: np.sum((x-y)**2)
        name_statistic = "euclidean distance"

    if os.path.exists(bayesian_path + "/ns_result.csv"):
        df_bayes = pd.read_csv(bayesian_path + "/ns_result.csv", index_col=0)

    if os.path.exists(mc_path + "/mc_result.csv"):
        df_mc = pd.read_csv(mc_path + "/mc_result.csv", index_col=0)

    x_bayes = df_bayes.values
    y_mc = df_mc.values
    parameter_names = df_mc.iloc[0].index.values

    res = permutation_test((x_bayes, y_mc), **perm_settings)

    null_distribution = res.null_distribution

    p_value = res.pvalue

    for i, param in enumerate(parameter_names):
        fig, ax = ptl.subplots()
        ax.hist(null_distribution[:, i], bins=50, label=f"Null Distribution {param}")
        ax.axvline(
            res.statistic[i], color="red", linestyle="--", label=f"Statistic {param}"
        )
        ax.set_title(
            f"Statistic: {name_statistic} \n Permutation test p-value: {p_value[i]:.2f}"
        )
        ax.legend()
        yield fig


if __name__ == "__main__":
    import pandas as pd

    bayes_fit = "grid_pdf_bayes_numerical"
    mc_fit = bayes_fit
    # df_bayes = pd.read_csv(bayesian_path + "/ns_result.csv", index_col=0)
    # df_mc = pd.read_csv(mc_path + "/ns_result.csv", index_col=0)
    # x = df_bayes.values
    # y = df_mc.values

    mean_stat = lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis)
    var_stat = lambda x, y, axis: np.var(x, axis=axis) - np.var(y, axis=axis)

    perm_settings = {
        "n_resamples": 1000,
        "alternative": "two-sided",
        "statistic": "mean",
    }

    plot_perm_test(mc_fit=mc_fit, bayesian_fit=bayes_fit, perm_settings=perm_settings)
