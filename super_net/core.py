import jax
import jax.numpy as jnp
import itertools

from validphys import convolution
from validphys.fkparser import load_fktable
from validphys.core import DataSetSpec, DataGroupSpec

from super_net.theory_predictions import make_dis_prediction, make_had_prediction


# Is this needed? -> probably no need to jit compile
OP = {key: jax.jit(val) for key,val in convolution.OP.items()}


class NewDataSetSpec(DataSetSpec):
    """
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
            pred = make_dis_prediction(fk)
            pred_funcs.append(pred)
            
        @jax.jit
        def prediction(pdf):
            return OP[self.op](*[f(pdf) for f in pred_funcs])
        
        return prediction


class NewDataGroupSpec(DataGroupSpec):
    """
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
            predictions.append(ds.make_pred_dataset)
            
        @jax.jit
        def eval_preds(pdf):
            return jnp.array(list(itertools.chain(*[f(pdf) for f in predictions])))

        return eval_preds