"""
colibri.coredata

Module containing validphys.coredata data structures with some additional methods.
"""

import jax.numpy as jnp
import dataclasses

from validphys.coredata import FKTableData


@dataclasses.dataclass(eq=False)
class FKTableData(FKTableData):
    """
    Inherits from validphys.coredata.FKTableData and has additional methods.
    """

    def with_masked_flavours(self, flavour_indices):
        """
        Method that replaces the FKTableData instance with a new one with masked flavours,
        both in the luminosity mapping and in the FK array.

        Parameters
        ----------
        flavour_indices: list
            The indices of the flavours to keep.

        """
        if flavour_indices is None:
            return self

        if self.hadronic:
            lumi_indices = self.luminosity_mapping
            mask_even = jnp.isin(lumi_indices[0::2], jnp.array(flavour_indices))
            mask_odd = jnp.isin(lumi_indices[1::2], jnp.array(flavour_indices))

            # for hadronic predictions pdfs enter in pair, hence product of two
            # boolean arrays and repeat by 2
            mask = jnp.repeat(mask_even * mask_odd, repeats=2)
            lumi_indices = lumi_indices[mask]

            fk_arr_mask = mask_even * mask_odd

            fk_array = self.get_np_fktable().copy()

            # replace the luminosity indices with the masked ones
            new_instance = dataclasses.replace(self, luminosity_mapping=lumi_indices)

            # replace the FK array with the masked one
            return dataclasses.replace(
                new_instance, get_np_fktable=fk_array[:, fk_arr_mask, :, :]
            )

        else:

            lumi_indices = self.luminosity_mapping
            mask = jnp.isin(lumi_indices, jnp.array(flavour_indices))
            lumi_indices = lumi_indices[mask]

            fk_array = self.get_np_fktable().copy()

            new_instance = dataclasses.replace(self, luminosity_mapping=lumi_indices)

            return dataclasses.replace(
                new_instance, get_np_fktable=fk_array[:, mask, :]
            )
