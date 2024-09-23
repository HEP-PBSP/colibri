"""
wmin.completeness.py

This module contains the functions for checking the completeness of a weight-minimisation
basis. It includes plotting tools.
"""

from validphys.core import PDF
from reportengine.figure import figure, figuregen

import matplotlib.pyplot as plt

import numpy as np


def basis_to_target_distances(
    target_pdfs,
    wmin_basis_pdfs,
    nweights,
    pdf_aliases,
    Q=1.65,
    x_grid=np.logspace(-5, 0, 50),
    flavours=[1, -1, 2, -2, 3, -3, 4, 21],
):
    """Computes the distances from the target PDFs to the wmin_basis_pdfs."""

    target_pdf_sets = []
    for target_pdf in target_pdfs:
        name = pdf_aliases.get(target_pdf, target_pdf)
        target_pdf_sets.append(
            {"name": name, "members": PDF(target_pdf).load().members[1:]}
        )

    wmin_bases = []
    for pdf_basis in wmin_basis_pdfs:
        name = pdf_aliases.get(pdf_basis, pdf_basis)
        loaded_pdf = PDF(pdf_basis).load()
        wmin_bases.append(
            {
                "name": name,
                "central": pdf_grid_allflav(
                    loaded_pdf.central_member, flavours, x_grid, Q
                ),
                "members": loaded_pdf.members[1 : nweights + 1],
            }
        )

    distances = {}
    for basis in wmin_bases:
        distances[basis["name"]] = []
        distance = []
        for pdf in target_pdf_sets:
            for member in pdf["members"]:
                original, reco, w, d = wmin_distance(
                    member,
                    basis["central"],
                    basis["members"],
                    flavours,
                    x_grid,
                    Q,
                    dist_type=0,
                )
                distance.append(d)
            distances[basis["name"]].append(
                {"name": pdf["name"], "distances": distance}
            )

    return distances


@figuregen
def histogram_individual_target_sets(basis_to_target_distances):
    """Produces plots showing the error in the approximation when the
    wmin_basis_pdf is used to approximate the target_pdfs.
    """

    for basis, targets in basis_to_target_distances.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        overflow_threshold = 0.05
        bins = np.linspace(0, overflow_threshold, 30)

        for index, target in enumerate(targets):
            d = np.array(target["distances"])
            d[d > overflow_threshold] = overflow_threshold + 0.001
            targets[index] = {"name": target["name"], "distances": d}

        ax.hist(
            [target["distances"] for target in targets],
            label=[target["name"] for target in targets],
            # alpha=0.6,
            bins=np.append(bins, 0.052),
            # edgecolor="black",
            linewidth=2,
            histtype="step",
            density=False,
            stacked=True,
        )
        ax.set_title("Distances of PDF sets from %s " % basis)
        ax.set_xlabel("Score", fontsize=16)
        ax.set_ylabel("Count", fontsize=16)
        ax.legend(frameon=False, fontsize=14)
        # set xticks to specific values
        ax.set_xticks([0, 0.01, 0.02, 0.03, 0.04])
        # add a tick after 0.05, overflow, with label > 0.5
        ax.set_xticks([0.051], minor=True)
        ax.set_xticklabels(["$>$0.05"], minor=True)

        yield fig


@figure
def histogram_complete_target_set(basis_to_target_distances):
    """Produces a plot showing the error in the approximation when
    the wmin_basis_pdf is used to approximate the target_pdfs.
    """

    fig, ax = plt.subplots(figsize=(7, 5))
    overflow_threshold = 0.05
    bins = np.linspace(0, overflow_threshold, 30)
    for basis, targets in basis_to_target_distances.items():
        all_distances = []
        for target in targets:
            all_distances.extend(target["distances"])
        d = np.array(all_distances)
        d[d > overflow_threshold] = overflow_threshold + 0.001
        ax.hist(
            d,
            label="Basis: %s" % basis,
            # alpha=0.6,
            bins=np.append(bins, 0.052),
            # edgecolor="black",
            linewidth=2,
            histtype="step",
            density=False,
        )
    ax.set_title("Distances of weight-minimisation bases to target set")
    ax.set_xlabel("Score", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)
    ax.legend(frameon=False, fontsize=14)
    # set xticks to specific values
    ax.set_xticks([0, 0.01, 0.02, 0.03, 0.04])
    # add a tick after 0.05, overflow, with label > 0.5
    ax.set_xticks([0.051], minor=True)
    ax.set_xticklabels(["$>$0.05"], minor=True)

    return fig


def pdf_grid(pdf, pid, x_grid, Q):
    out = []
    for x in x_grid:
        out.append(pdf.xfxQ(pid, x, Q))
    return np.array(out)


def pdf_grid_allflav(pdf, flavours, x_grid, Q):
    out = np.array([])
    for pid in flavours:
        out = np.append(out, pdf_grid(pdf, pid, x_grid, Q))
    return out


def wmin_distance(
    pdf_target, center_pdf_grid, wmin_basis, flavours, x_grid, Q, dist_type=0
):
    Y = pdf_grid_allflav(pdf_target, flavours, x_grid, Q) - center_pdf_grid
    X = np.array(
        [
            pdf_grid_allflav(replica, flavours, x_grid, Q) - center_pdf_grid
            for replica in wmin_basis
        ]
    ).T

    if dist_type == 0:
        Sigma = np.identity(len(Y))

    elif dist_type == 1:
        Sigma = np.diag(np.abs(1 / (Y + center_pdf_grid)))

    elif dist_type == 2:
        Sigma = np.diag(np.abs(1 / (Y + center_pdf_grid) ** 2))

    elif dist_type == 3:
        X_grid = np.tile(x_grid, len(flavours))
        Sigma = np.diag(np.abs(X_grid / (Y + center_pdf_grid)))

    elif dist_type == 4:
        X_grid = np.tile(x_grid, len(flavours))
        Sigma = np.diag(np.abs(X_grid))

    w = np.linalg.inv(X.T @ Sigma @ X) @ X.T @ Sigma @ Y
    distance = (Y - X @ w) @ Sigma @ (Y - X @ w)

    original = pdf_grid_allflav(pdf_target, flavours, x_grid, Q).reshape(
        len(flavours), len(x_grid)
    )
    reco = (center_pdf_grid + X @ w).reshape(len(flavours), len(x_grid))

    return original, reco, w, distance
