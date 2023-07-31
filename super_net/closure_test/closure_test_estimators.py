import numpy as np
import pandas as pd

from validphys import covmats
from validphys.calcutils import calc_chi2

from reportengine.table import table


@table
def table_closure_test_central_chi2(data, closure_test_pdf, pdf):
    """
    Computes the chi2 between the central theory prediction of the underlying
    law (`closure_test_pdf`) used in the closure test and the central theory
    prediction of a given PDF set (`pdf`).
    The covariance matrix used is the t0 covariance matrix with t0 predictions
    taken from the `closure_test_pdf`

    Parameters
    ----------

    data: core.SuperNetDataGroupSpec

    closure_test_pdf: core.PDF

    pdf: core.PDF

    Returns
    -------
    float
        chi2 computed as described above.

    """

    cd_list_ct = data.load_pseudo_commondata(
        closure_test_pdf=closure_test_pdf, fakedata=True
    )
    cv_ct = np.array([cd.central_values.to_numpy() for cd in cd_list_ct])

    cd_list_pdf = data.load_pseudo_commondata(closure_test_pdf=pdf, fakedata=True)
    cv_pdf = np.array([cd.central_values.to_numpy() for cd in cd_list_pdf])

    # compute chi2 for each dataset
    records = []
    for cd_ct, cd_pdf, ds in zip(cd_list_ct, cd_list_pdf, data.datasets):
        diff = cd_ct.central_values - cd_pdf.central_values
        t0_ds_covmat = covmats.t0_covmat_from_systematics(
            cd_ct,
            dataset_input=ds,
            use_weights_in_covmat=False,
            norm_threshold=None,
            dataset_t0_predictions=cd_ct.central_values,
        )
        sqrt_t0_ds_covmat = covmats.sqrt_covmat(t0_ds_covmat)

        ds_chi2 = calc_chi2(sqrt_t0_ds_covmat, diff)

        records.append(
            dict(
                dataset=str(ds),
                chi2=ds_chi2,
                ndata=cd_ct.ndata,
                chi2_norm=ds_chi2 / cd_ct.ndata,
            )
        )

    # compute total chi2
    diffs = np.concatenate(cv_ct) - np.concatenate(cv_pdf)
    t0_covmat = covmats.dataset_inputs_t0_covmat_from_systematics(
        cd_list_ct,
        data_input=data.dsinputs,
        use_weights_in_covmat=False,
        norm_threshold=None,
        dataset_inputs_t0_predictions=cv_ct,
    )
    sqrt_t0_covmat = covmats.sqrt_covmat(t0_covmat)
    total_chi2 = calc_chi2(sqrt_t0_covmat, diffs)

    records.append(
        dict(
            dataset="total",
            chi2=total_chi2,
            ndata=len(diffs),
            chi2_norm=total_chi2 / len(diffs),
        )
    )

    df = pd.DataFrame.from_records(records, index="dataset")
    df.columns = ["chi2", "ndata", "chi2/ndata"]
    return df
