import jax
import jax.numpy as jnp


def make_dis_prediction(fk):
    """
    given an FKTableData instance returns a jax.jit
    compiled function taking a pdf grid as input
    and returning a theory prediction for a DIS
    observable.
    
    Parameters
    ----------
    fk : validphys.coredata.FKTableData
    
    Returns
    -------
    @jax.jit CompiledFunction 
    """
    indices = fk.luminosity_mapping
    fk_arr = jnp.array(fk.get_np_fktable())
    
    @jax.jit
    def dis_prediction(pdf):
        return jnp.einsum("ijk, jk ->i", fk_arr, pdf[indices, :])

    return dis_prediction


def make_had_prediction(fk):
    """
    given an FKTableData instance returns a jax.jit
    compiled function taking a pdf grid as input
    and returning a theory prediction for a hadronic
    observable.
    
    Parameters
    ----------
    fk : validphys.coredata.FKTableData
    
    Returns
    -------
    @jax.jit CompiledFunction 
    """

    indices = fk.luminosity_mapping
    first_indices = indices[0::2]
    second_indices = indices[1::2]
    fk_arr = jnp.array(fk.get_np_fktable())
    @jax.jit
    def had_prediction(pdf):
        return jnp.einsum("ijkl,jk,jl->i", fk_arr, pdf[first_indices,:], pdf[second_indices,:])
    return had_prediction