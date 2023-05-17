"""
Module containing similar classes or classes that inherit from those in `validphis.coredata`
"""

from validphys.coredata import FKTableData
import numpy as np
from scipy.interpolate import interp1d, griddata
import pandas as pd

import dataclasses


class NewFKTableData(FKTableData):
    """
    Inherits from validphys.FKTableData dataclass
    """

    # this is very ugly but works for the moment
    @property
    def xgrid_new(self):
        return np.array(
            [
                2.00000000e-07,
                3.03430477e-07,
                4.60350147e-07,
                6.98420853e-07,
                1.05960950e-06,
                1.60758550e-06,
                2.43894329e-06,
                3.70022721e-06,
                5.61375772e-06,
                8.51680668e-06,
                1.29210157e-05,
                1.96025050e-05,
                2.97384954e-05,
                4.51143839e-05,
                6.84374492e-05,
                1.03811730e-04,
                1.57456056e-04,
                2.38787829e-04,
                3.62054496e-04,
                5.48779532e-04,
                8.31406884e-04,
                1.25867971e-03,
                1.90346340e-03,
                2.87386758e-03,
                4.32850064e-03,
                6.49620619e-03,
                9.69915957e-03,
                1.43750686e-02,
                2.10891867e-02,
                3.05215840e-02,
                4.34149174e-02,
                6.04800288e-02,
                8.22812213e-02,
                1.09143757e-01,
                1.41120806e-01,
                1.78025660e-01,
                2.19504127e-01,
                2.65113704e-01,
                3.14387401e-01,
                3.66875319e-01,
                4.22166775e-01,
                4.79898903e-01,
                5.39757234e-01,
                6.01472198e-01,
                6.64813948e-01,
                7.29586844e-01,
                7.95624252e-01,
                8.62783932e-01,
                9.30944081e-01,
                1.00000000e00,
            ]
        )

    @property
    def dis_sigma(self):
        """
        Given a DIS FKTableData instance, interpolate the columns
        of the old sigma table vs the xgrid. The interpolator
        function is then used to compute fktable values on the new xgrid.

        Notes:
        - `scipy.interpolate.interp1d` is used for the interpolation
        - linear spline interpolation is used (`kind='slinear'`)
        - values in extrapolation region are set to zero

        Returns
        -------
        pd.DataFrame
            multiindex pandas dataframe corresponding to
            DIS sigma table
        """

        xgrid = self.xgrid
        xgrid_new = self.xgrid_new

        dfs = []
        # group over datapoints
        for d, grp in self.sigma.groupby("data"):
            x_vals = xgrid[grp.index.get_level_values("x")]

            interpolators = {
                col: interp1d(
                    x_vals,
                    grp[col].values,
                    kind="slinear",
                    fill_value=0,
                    bounds_error=False,
                )
                for col in grp.columns
            }

            col_dict = dict()
            for col in grp.columns:
                col_dict[f"{col}"] = interpolators[col](xgrid_new)

            # generate multiindex dataframe
            tmp_index = pd.MultiIndex.from_product(
                [[d], range(len(xgrid_new))], names=["data", "x"]
            )
            tmp_df = pd.DataFrame(col_dict, index=tmp_index)
            dfs.append(tmp_df)

        return pd.concat(dfs, axis=0)

    @property
    def had_sigma(self):
        """
        Given a hadronic FKTableData instance, perform 2D interpolation of
        columns vs (x1,x2) grid. The interpolator function is then used to
        compute fktable values on the new xgrid.

        Notes:
        - `scipy.interpolate.griddata` is used to interpolate
        - 'nearest' method is used, this has to be followed by setting the
          interpolated function to zero in the extrapolation region


        Returns
        -------
        pd.DataFrame
            multiindex pandas dataframe corresponding to the
            hadronic sigma table
        """
        xgrid = self.xgrid
        xgrid_new = self.xgrid_new

        new_xgrid_mesh = np.meshgrid(xgrid_new, xgrid_new)

        # Flatten the grid into two separate 1D arrays
        new_x1grid_flat = np.ravel(new_xgrid_mesh[0])
        new_x2grid_flat = np.ravel(new_xgrid_mesh[1])

        dfs = []
        # group by datapoints
        for d, grp in self.sigma.groupby("data"):
            x1_vals = xgrid[grp.index.get_level_values("x1")]
            x2_vals = xgrid[grp.index.get_level_values("x2")]

            interpolated_grids = {
                col: griddata(
                    points=(x1_vals, x2_vals),
                    values=grp[col].values,
                    xi=(new_x1grid_flat, new_x2grid_flat),
                    method="nearest",
                )
                for col in grp.columns
            }

            ################### NOT SURE WHETHER THIS IS CORRECT ############################
            # set to zero in extrapolation regions
            smaller_than = set(np.where(new_x1grid_flat < np.min(x1_vals))[0])
            larger_than = set(np.where(new_x1grid_flat > np.max(x1_vals))[0])
            set_to_zero = np.array(list(larger_than.union(smaller_than)))
            #################################################################################

            col_dict = dict()
            for col in grp.columns:
                interpolated_grids[col][set_to_zero] = 0
                col_dict[f"{col}"] = interpolated_grids[col]

            tmp_index = pd.MultiIndex.from_product(
                [[d], range(len(xgrid_new)), range(len(xgrid_new))],
                names=["data", "x1", "x2"],
            )
            tmp_df = pd.DataFrame(col_dict, index=tmp_index)
            dfs.append(tmp_df)

        return pd.concat(dfs, axis=0)

    @property
    def new_sigma(self):
        if self.hadronic:
            return self.had_sigma
        else:
            return self.dis_sigma

    def with_new_sigma(self, new_sigma):
        return dataclasses.replace(self, sigma=new_sigma)
