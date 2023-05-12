"""
Module containing similar classes or classes that inherit from those in `validphis.core`
"""

from validphys.core import DataGroupSpec
import numpy as np
import numpy.random as npr

class NewDataGroupSpec(DataGroupSpec):
    """
    Inherits from validphys.DataGroupSpec
    """

            
    def data_batch_stream_index(self, batch_size=128, seed=0):
        """ yields array of indices of specified shape.
        The indices are chosen at random via shuffling.
        
        Parameters
        ----------
        batch_size : 
        
        seed : {None, int, array_like}, optional, default = 0
                Random seed used to initialize the pseudo-random number generator. 
                Can be any integer between 0 and 2**32 - 1 inclusive, 
                an array (or other sequence) of such integers, or None.
                
        """
        
        Ndat = np.sum([cd.ndata for cd in self.load_commondata_instance()])
        
        if batch_size > Ndat:
            raise ValueError(f"size of batch = {batch_size} should be smaller or equal to the number of data {Ndat}")
        
        rng = npr.RandomState(seed)
        num_complete_batches, leftover = divmod(Ndat, batch_size)
        # discard leftover to avoid the a slow down due to having to recompile make_chi2 functionm
        print(f"Ndat = {Ndat}, for each epoch leftover = {leftover}")
        
        num_batches = num_complete_batches #+ bool(leftover)

        while True:
            perm = rng.permutation(Ndat)
            
            for i in range(num_batches):
                batch_idx = perm[i*batch_size:(i+1)*batch_size]
                yield batch_idx, num_batches

    
        
        
        
