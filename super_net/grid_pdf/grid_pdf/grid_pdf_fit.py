from grid_pdf.grid_pdf_model import FLAVOUR_MAPPING
from validphys.convolution import FK_FLAVOURS

import ultranest
import jax
import time
import logging

log = logging.getLogger(__name__)


def make_bayesian_pdf_grid_fit(
    make_chi2_with_positivity,
    grid_pdf_model_prior,
    interpolate_grid,
    reduced_xgrids,
    flavour_mapping=FLAVOUR_MAPPING,
    min_num_live_points=400,
    min_ess=40,
    log_dir="ultranest_logs",
    resume=True,
    vectorised=False,
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
        return -0.5 * make_chi2_with_positivity(pdf)

    parameters = [
        f"{FK_FLAVOURS[i]}({j})" for i in flavour_mapping for j in reduced_xgrids[i]
    ]

    sampler = ultranest.ReactiveNestedSampler(
        parameters,
        log_likelihood,
        grid_pdf_model_prior,
        log_dir=log_dir,
        resume=resume,
        vectorized=vectorised,
        ndraw_max=ndraw_max,
    )

    if slice_sampler:
        import ultranest.stepsampler as ustepsampler

        sampler.stepsampler = ustepsampler.SliceSampler(
            nsteps=slice_steps,
            generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
        )

    t0 = time.time()
    sampler.run(
        min_num_live_points=min_num_live_points,
        min_ess=min_ess,
    )
    t1 = time.time()
    log.info("ULTRANEST RUNNING TIME: %f" % (t1 - t0))

    # Store run plots to ultranest output folder
    sampler.plot()
