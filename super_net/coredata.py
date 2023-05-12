"""
Module containing similar classes or classes that inherit from those in `validphis.coredata`
"""

from validphys.coredata import FKTableData
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd


class NewFKTableData(FKTableData):
    """ """

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
        Given a DIS FKTableData instance, interpolate (using
        linear splines) the columns of the old sigma table vs
        the xgrid. The interpolator function is then used to 
        compute / extrapolate fktable values on the new xgrid 


        Returns
        -------
        pd.DataFrame
            multiindex pandas dataframe where first index is datapoint
            and second index is x

        """

        xgrid = self.xgrid
        xgrid_new = self.xgrid_new

        dfs = []
        # group over datapoints
        for d, grp in self.sigma.groupby(self.sigma.index.get_level_values("data")):
            interpolators = [
                interp1d(
                    xgrid[grp.index.get_level_values("x")],
                    grp[col],
                    fill_value="extrapolate",
                )
                for col in grp.columns
            ]

            d = dict(
                data=tuple(
                    np.array(
                        np.ones(50) * grp.index.get_level_values("data").unique(),
                        dtype=int,
                    )
                ),
                x=tuple(np.arange(0, 50)),
            )

            for col, interp in zip(grp.columns, interpolators):
                d[f"{col}"] = interp(xgrid_new)

            dfs.append(pd.DataFrame(d))

        new_sigma = pd.concat(dfs, axis=0)
        new_sigma = (
            new_sigma.reset_index().set_index(["data", "x"]).drop(["index"], axis=1)
        )

        return new_sigma

    @property
    def had_sigma(self):
        """ 
        Given a hadronic FKTableData instance, interpolate (using
        linear splines) the columns of the old sigma table vs
        the xgrid. The interpolator function is then used to 
        compute / extrapolate fktable values on the new xgrid 
        

        Returns
        -------
        pd.DataFrame
            multiindex pandas dataframe where first index is datapoint
            and second index is x

        """
        xgrid = self.xgrid
        xgrid_new = self.xgrid_new

        dfs = []

        # group by datapoints and x1
        for d, grp in self.sigma.groupby(["data", "x1"]):
            interpolators = [
                interp1d(
                    xgrid[grp.index.get_level_values("x2")],
                    grp[col],
                    fill_value="extrapolate",
                )
                for col in grp.columns
            ]

            d = dict(
                data=tuple(
                    np.array(
                        np.ones(50) * grp.index.get_level_values("data").unique(),
                        dtype=int,
                    )
                ),
                x1=tuple(np.arange(0, 50)),
                x2=tuple(np.arange(0, 50)),
            )

            for col, interp in zip(grp.columns, interpolators):
                d[f"{col}"] = interp(xgrid_new)

            dfs.append(pd.DataFrame(d))

        new_sigma = pd.concat(dfs, axis=0)
        new_sigma = (
            new_sigma.reset_index()
            .set_index(["data", "x1", "x2"])
            .drop(["index"], axis=1)
        )
        return new_sigma

    @property
    def new_sigma(self):
        if self.hadronic:
            return self.had_sigma
        else:
            return self.dis_sigma
