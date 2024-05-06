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
from flax.training.early_stopping import EarlyStopping
import pandas as pd
import torch
from reportengine import colors

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(colors.ColorHandler())

torch.set_default_dtype(torch.float64)


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
    parser.add_argument(
        "--verbose",
        type=int,
        default=500,
        help="Print loss every n epochs, default 500",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--min_delta", type=float, default=1e-5, help="Early stopping minimum delta"
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.8,
        help="Fraction of data to use for training, default 0.8",
    )

    args = parser.parse_args()

    # set random seed for reproducibility
    torch.manual_seed(args.seed)

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

    # split the data into training and validation sets
    n_train = int(args.train_frac * target_dist.shape[0])
    target_dist_train = target_dist[:n_train]
    target_dist_val = target_dist[n_train:]

    # test that hidden_dim is larger or equal to data_dim
    if args.hidden_dim < data_dim:
        raise ValueError(
            f"hidden_dim must be larger or equal to data_dim. Got hidden_dim={args.hidden_dim} and data_dim={data_dim}"
        )

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

    early_stopper = EarlyStopping(min_delta=args.min_delta, patience=args.patience)

    for idx in range(args.nepochs):
        opt.zero_grad()

        # Minimize KL(p || q)
        loss = -flow.log_prob(target_dist_train).mean()

        # evaluate on validation set
        val_loss = -flow.log_prob(target_dist_val).mean()

        # use stopping criterion based on validation loss
        early_stopper = early_stopper.update(val_loss)
        if early_stopper.should_stop:
            log.info("Met early stopping criteria, breaking...")
            break

        if idx % args.verbose == 0:
            log.info(
                f"epoch, {idx}, loss, {loss / data_dim:.4f}, val_loss, {val_loss / data_dim:.4f}"
            )

        loss.backward()
        opt.step()

    # save the flow in the fit directory using dill
    with open(fit_path / "norm_flow.pkl", "wb") as file:
        dill.dump(flow, file)
