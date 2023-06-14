import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import itertools

import pandas as pd
from sklearn.model_selection import train_test_split

from validphys import convolution
from validphys.fkparser import load_fktable
from validphys.core import DataSetSpec, DataGroupSpec
from validphys import covmats

from super_net.theory_predictions import (
    make_dis_prediction,
    make_had_prediction,
    make_pred_dataset,
)
from super_net.covmats import sqrt_covmat_jax

# Is this needed? -> probably no need to jit compile
OP = {key: jax.jit(val) for key, val in convolution.OP.items()}


class NewDataSetSpec(DataSetSpec):
    """
    Class Inheriting from `validphys.core.DataSetSpec`
    """

    @property
    def make_pred_dataset(self):
        """
        Compute theory prediction for a DataSetSpec

        Returns
        -------
        @jax.jit CompiledFunction
            Compiled function taking pdf grid in input
            and returning theory prediction for one
            dataset
        """

        pred_funcs = []

        for fkspec in self.fkspecs:
            fk = load_fktable(fkspec).with_cuts(self.cuts)
            if fk.hadronic:
                pred = make_had_prediction(fk)
            else:
                pred = make_dis_prediction(fk)
            pred_funcs.append(pred)

        @jax.jit
        def prediction(pdf):
            return OP[self.op](*[f(pdf) for f in pred_funcs])

        return prediction


class NewDataGroupSpec(DataGroupSpec):
    """
    Class inheriting from `validphys.core.DataGroupSpec`
    """

    @property
    def make_pred_data(self):
        """
        Compute theory prediction for a DataGroupSpec

        Returns
        -------
        @jax.jit CompiledFunction
            Compiled function taking pdf grid in input
            and returning theory prediction for one
            data group
        """

        predictions = []
        for ds in self.datasets:
            predictions.append(make_pred_dataset(ds))

        @jax.jit
        def eval_preds(pdf):
            return jnp.array(list(itertools.chain(*[f(pdf) for f in predictions])))

        return eval_preds

    @property
    def data_values(self):
        """
        Method of NewDataGroupSpec.
        Used to get data values, covariance matrix,
        and indices of data values.

        Returns
        -------
        tuple
            3D tuple containing
            - jnp.ndarray of central values of data
            - jnp.ndarray for experimental covariance matrix of data
            - jnp.ndarray of indices of data values
        """

        cd_list = self.load_commondata_instance()

        central_values = [cd.central_values for cd in cd_list]

        central_values = jnp.array(pd.concat(central_values, axis=0))

        covmat = jnp.array(
            covmats.dataset_inputs_covmat_from_systematics(
                cd_list, data_input=self.dsinputs
            )
        )

        indices = jnp.arange(central_values.shape[0])

        return central_values, covmat, indices

    def train_validation_split(self, test_size=0.2, random_state=42):
        """
        Get training validation split for the data values.

        Parameters
        ----------
        test_size : float, default is 0.2
                size of the test/validation set, float between 0 and 1

        random_state : int, default is 42
                    integer specifiying the random state of the training
                    test split

        Returns
        -------
        tuple
            6D tuple containing:
            - jnp.ndarray of training central values
            - jnp.ndarray of training covmat
            - jnp.ndarray of training indices
            - jnp.ndarray of validation central values
            - jnp.ndarray of validation covmat
            - jnp.ndarray of validation indices
        """

        central_values, covmat, indices = self.data_values

        # Perform train-test split on indices
        indices_train, indices_val = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )

        # Use indices to split central values, covariance matrix and predictions
        central_values_train, central_values_val = (
            central_values[indices_train],
            central_values[indices_val],
        )
        covmat_train, covmat_val = (
            covmat[indices_train][:, indices_train],
            covmat[indices_val][:, indices_val],
        )

        return (
            central_values_train,
            covmat_train,
            indices_train,
            central_values_val,
            covmat_val,
            indices_val,
        )

    # @property
    def make_chi2_data_mini_batch(
        self, train_val_split=False, test_size=0.2, random_state=42
    ):
        """
        Compute the chi2 between experimental central values
        and theoretical predictions using the experimental
        covariance matrix

        Returns
        -------
        @jax.jit CompiledFunction
            Compiled function taking pdf grid, and indexes for
            the data batch (random subset of datapoints of the
            whole dataset) in input and returning the chi2
            between experimental central values and th. predictions
            using the exp covmat.
        """

        if train_val_split:
            (
                central_values_train,
                covmat_train,
                indices_train,
                _,
                _,
                _,
            ) = self.train_validation_split(
                test_size=test_size, random_state=random_state
            )
        else:
            central_values_train, covmat_train, indices_train = self.data_values

        pred = self.make_pred_data

        @jax.jit
        def chi2(pdf, batch_idx):
            """
            Compute batched chi2

            Parameters
            ----------
            pdf :

            batch_idx :

            """
            diff = pred(pdf)[indices_train][batch_idx] - central_values_train[batch_idx]

            # batch covariance matrix before decomposing it
            batched_covmat = covmat_train[batch_idx][:, batch_idx]
            # decompose covmat after having batched it!
            sqrt_covmat = jnp.array(sqrt_covmat_jax(batched_covmat))

            # solve_triangular: solve the equation a x = b for x, assuming a is a triangular matrix.
            chi2_vec = jla.solve_triangular(sqrt_covmat, diff, lower=True)
            return jnp.sum(chi2_vec**2)

        return chi2

    def make_chi2_validation(self, test_size=0.2, random_state=42):
        """
        Compute the chi2 on the validation set. The chi2 is
        computed between experimental central values
        and theoretical predictions using the experimental
        covariance matrix.

        Returns
        -------
        @jax.jit CompiledFunction
            Compiled function taking pdf grid in input and returning the chi2
            between experimental central values and th. predictions
            using the exp covmat.
        """

        (
            _,
            _,
            _,
            central_values_val,
            covmat_val,
            indices_val,
        ) = self.train_validation_split(test_size=test_size, random_state=random_state)

        pred = self.make_pred_data

        @jax.jit
        def chi2(pdf):
            """
            Compute batched chi2

            Parameters
            ----------
            pdf :

            """
            diff = pred(pdf)[indices_val] - central_values_val

            sqrt_covmat = jnp.array(sqrt_covmat_jax(covmat_val))

            # solve_triangular: solve the equation a x = b for x, assuming a is a triangular matrix.
            chi2_vec = jla.solve_triangular(sqrt_covmat, diff, lower=True)
            return jnp.sum(chi2_vec**2)

        return chi2
