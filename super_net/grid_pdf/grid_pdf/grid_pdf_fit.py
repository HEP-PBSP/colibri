from grid_pdf.grid_pdf_model import FLAVOUR_MAPPING
from validphys.convolution import FK_FLAVOURS

from grid_pdf.grid_pdf_lhapdf import lhapdf_grid_pdf_ultranest_result
from super_net.ns_utils import resample_from_ns_posterior

from validphys.loader import Loader
from validphys.lhio import generate_replica0

import ultranest
import jax
import time
import logging

log = logging.getLogger(__name__)

def ultranest_grid_fit(
    _chi2_with_positivity,
    grid_pdf_model_prior,
    interpolate_grid,
    reduced_xgrids,
    flavour_indices,
    min_num_live_points=400,
    min_ess=40,
    log_dir="ultranest_logs",
    resume=True,
    vectorized=False,
    slice_sampler=False,
    slice_steps=100,
    ndraw_max=500,
):
    """
    TODO

    Parameters
    ----------

    Returns
    -------

    """

    @jax.jit
    def log_likelihood(stacked_pdf_grid):
        """
        TODO

        Parameters
        ----------
        stacked_pdf_grid: jnp.array

        Returns
        -------

        """

        pdf = interpolate_grid(stacked_pdf_grid)
        return -0.5 * _chi2_with_positivity(pdf)

    parameters = [
        f"{FK_FLAVOURS[i]}({j})" for i in flavour_indices for j in reduced_xgrids[i]
    ]

    sampler = ultranest.ReactiveNestedSampler(
        parameters,
        log_likelihood,
        grid_pdf_model_prior,
        log_dir=log_dir,
        resume=resume,
        vectorized=vectorized,
        ndraw_max=ndraw_max,
    )

    if slice_sampler:
        import ultranest.stepsampler as ustepsampler

        sampler.stepsampler = ustepsampler.SliceSampler(
            nsteps=slice_steps,
            generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
        )

    t0 = time.time()
    ultranest_result = sampler.run(
        min_num_live_points=min_num_live_points,
        min_ess=min_ess,
    )
    t1 = time.time()
    log.info("ULTRANEST RUNNING TIME: %f" % (t1 - t0))

    if n_posterior_samples > ultranest_result["samples"].shape[0]:
        n_posterior_samples = ultranest_result["samples"].shape[0] - int(
            0.1 * ultranest_result["samples"].shape[0]
        )
        log.warning(
            f"The chosen number of posterior samples exceeds the number of posterior"
            "samples computed by ultranest. Setting the number of resampled posterior"
            f"samples to {n_posterior_samples}"
        )

    resampled_posterior = resample_from_ns_posterior(
        ultranest_result["samples"],
        n_posterior_samples,
        posterior_resampling_seed,
    )

    # Store run plots to ultranest output folder
    sampler.plot()

    return resampled_posterior

def perform_nested_sampling_grid_pdf_fit(
    ultranest_grid_fit,
    reduced_xgrids,
    length_reduced_xgrids,
    n_posterior_samples,
    grid_pdf_fit_name,
    lhapdf_path,
    output_path,
    theoryid,
):
    """
    Performs a Nested Sampling fit using the grid.
    """

    # Produce the LHAPDF grid
    lhapdf_grid_pdf_ultranest_result(
        ultranest_grid_fit,
        reduced_xgrids,
        length_reduced_xgrids,
        n_posterior_samples,
        theoryid,
        grid_pdf_fit_name,
        folder=lhapdf_path,
        output_path=output_path,
    )

    # Produce the central replica
    l = Loader()
    pdf = l.check_pdf(grid_pdf_fit_name)
    generate_replica0(pdf)

    log.info("Nested Sampling grid PDF fit completed!")
