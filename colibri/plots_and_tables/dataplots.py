import numpy as np
from colibri.constants import XGRID
from colibri.plots_and_tables.plotting import ColibriFitsPlotter
from colibri.theory_predictions import make_pred_dataset
from reportengine.figure import figuregen


@figuregen
def plot_data_theory(colibri_fit, commondata_tuple, data, flavour_indices=None):
    """

    Parameters
    ----------
    colibri_fit: str
        Name of the colibri fit for which to compare data and theory.
        Needed for the theory part of the comparison.

    commondata_tuple: tuple, config.produce_commondata_tuple
        Needed for the data part of the comparison.

    data:
    """

    colibri_plotter = ColibriFitsPlotter(
        colibri_fit,
    )

    pdf_model = colibri_plotter.pdf_model

    grid_values_func = pdf_model.grid_values_func(XGRID)

    pdf_replicas = [
        grid_values_func(
            params=colibri_plotter.posterior_from_csv.values[replica_index, :]
        )
        for replica_index in range(colibri_plotter.posterior_from_csv.shape[0])
    ]

    for ds, cd in zip(data.datasets, commondata_tuple):
        ds_pred = make_pred_dataset(
                    ds, vectorized=False, flavour_indices=flavour_indices
                )
        theory_replicas = np.array(
            [
                ds_pred(pdf)
                for pdf in pdf_replicas
            ]
        )

        fig = colibri_plotter.plot_data_theory(
            ds,
            cd,
            theory_replicas,
        )

        yield fig
