"""
Script to learn the posterior distribution using a normalizing flow.
Uses a DenseAutoregressive flow with a standard normal base distribution.
"""

import argparse
import logging
import pathlib

import dill
import flowtorch.bijectors as bij
import flowtorch.distributions as dist
import flowtorch.parameters as params
import pandas as pd
import torch
from reportengine import colors

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(colors.ColorHandler())


def main():
    parser = argparse.ArgumentParser(
        description="Script to learn the posterior distribution using a normalizing flow."
    )
    parser.add_argument(
        "fit_name",
        help="Name of the fit for which to learn the posterior distribution using a normalizing flow.",
    )
    parser.add_argument("--nepochs", type=int, default=3001)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)

    args = parser.parse_args()

    # Convert fit_path to a pathlib.Path object
    fit_path = pathlib.Path(args.fit_name)

    # check whether fit path exists
    log.info(f"Training normalizing flow for {fit_path}")

    if not fit_path.exists():
        raise FileNotFoundError(f"Could not find the fit {fit_path}")

    # load target distribution: posterior samples from fit
    target_dist = torch.Tensor(
        pd.read_csv(fit_path / "full_posterior_sample.csv", index_col=0).to_numpy()
    )

    data_dim = target_dist.shape[1]

    # Lazily instantiated flow plus base and target distributions
    nn_params = params.DenseAutoregressive(hidden_dims=(args.hidden_dim,))

    bijectors = bij.AffineAutoregressive(params_fn=nn_params)

    # base distribution is a standard normal with dimension
    base_dist = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(data_dim), torch.ones(data_dim)), 1
    )

    # Instantiate transformed distribution and parameters
    flow = dist.Flow(base_dist, bijectors)

    # Training loop
    opt = torch.optim.Adam(flow.parameters(), lr=args.learning_rate)

    for idx in range(args.nepochs):
        opt.zero_grad()

        # Minimize KL(p || q)
        loss = -flow.log_prob(target_dist).mean()

        if idx % 500 == 0:
            log.info(f"epoch, {idx}, loss, {loss / data_dim:.4f}")

        loss.backward()
        opt.step()

    # save the flow in the fit directory using dill
    with open(fit_path / "norm_flow.pkl", "wb") as file:
        dill.dump(flow, file)
