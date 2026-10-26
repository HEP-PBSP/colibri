"""
colibri.pdf_model.py

This module implements an abstract class PDFModel which is filled by the various models.

"""

import logging
from abc import ABC, abstractmethod
from typing import Callable

import jax.numpy as jnp
from jax.typing import ArrayLike

log = logging.getLogger(__name__)


class PDFModel(ABC):
    """An abstract class describing the key features of a PDF model."""

    name = "Abstract PDFModel"

    #: Whether the fit writes the PDF to exportgrids, which evolve_fit turns
    #: into an LHAPDF set. Subclasses set it to False to skip this, e.g.
    #: FixedPDFModel, whose PDF is already an LHAPDF set.
    export_grids = True

    @property
    @abstractmethod
    def param_names(self) -> list:
        """This should return a list of names for the fitted parameters of the model.
        The order of the names is important as it will be assumed to be the order of the parameters
        fed to the model.
        """
        pass

    @property
    def n_parameters(self):
        """
        Returns the number of parameters of the pdf model.
        """
        return len(self.param_names)

    @abstractmethod
    def grid_values_func(self, xgrid: ArrayLike) -> Callable[[jnp.array], jnp.ndarray]:
        """This function should produce a grid values function, which takes
        in the model parameters, and produces the PDF values on the grid xgrid.
        The grid values function should be a function of the parameters and return
        an array of shape (N_fl, Nx). The first dimension is the number of flavours expected
        by the FK tables belonging to the chosen theoryID.
        The second dimension is the number of points in the xgrid, i.e. Nx = len(xgrid).

        Example
        -------
        ::

            def grid_values_func(xgrid):
                def func(params):
                    # Define expression for each flavour
                    fl_1 = params[0] + params[1] * xgrid
                    fl_2 = params[2] + params[3] * xgrid

                    # Combine the flavours into a single array
                    # This is just an example, the actual implementation will depend on the model
                    # and the number of flavours

                    return jnp.array([fl_1, fl_2])
                return func
        """
        pass


class FixedPDFModel(PDFModel):
    """
    A PDF model with no free parameters: one member of an LHAPDF set,
    evaluated at the parametrisation scale Q0.

    Used to fit only the extra parameters of a forward map (e.g. SMEFT Wilson
    coefficients) with the PDF fixed. Selected in the runcard with the
    ``fixed_pdf`` key.

    Parameters
    ----------
    pdf_name: str
        Name of the LHAPDF set.

    Q0: float
        Scale at which the PDF is evaluated.

    member: int, default 0
        Member of the set to use (0 is the central value).
    """

    name = "Fixed PDF model"

    # The PDF is already an LHAPDF set: there is nothing to export or evolve
    export_grids = False

    def __init__(self, pdf_name: str, Q0: float, member: int = 0):
        # Only plain data is stored so that the model can be pickled with dill
        self.pdf_name = str(pdf_name)
        self.Q0 = float(Q0)
        self.member = int(member)

    @property
    def param_names(self) -> list:
        return []

    def grid_values_func(self, xgrid: ArrayLike) -> Callable[[jnp.array], jnp.ndarray]:
        """The grid is computed once here; the returned function ignores params."""
        from validphys.core import PDF

        from colibri.utils import lhapdf_grid_values

        grid = lhapdf_grid_values(PDF(self.pdf_name), xgrid, self.Q0)[self.member]

        def pdf_func(params):
            return grid

        return pdf_func


def no_pdf_model():
    """Provider used when no PDF model is defined (base colibri, no ``fixed_pdf``)."""
    return None


def fixed_pdf_model(
    fixed_pdf,
    Q0,
    output_path,
    fixed_pdf_member=0,
    replica_index=None,
    positivity_penalty_settings=None,
    integrability_settings=None,
    dump_model=True,
):
    """
    Builds the :class:`FixedPDFModel` for the ``fixed_pdf`` LHAPDF set.

    Parameters
    ----------
    fixed_pdf: validphys.core.PDF

    Q0: float

    output_path: pathlib.Path
        The model is dumped here as ``pdf_model.pkl`` (needed by the scripts).

    fixed_pdf_member: int or "replica", default 0
        Member of the set to use. ``"replica"`` uses the MC replica index of the
        run (``-rep``), so that a set of Monte Carlo fits propagates the PDF
        uncertainty. Requires a Monte Carlo PDF set.

    replica_index: int, optional
        MC replica index from the command line.

    positivity_penalty_settings, integrability_settings: optional
        Only used to warn that the penalties are constant with a fixed PDF.

    dump_model: bool, default True
    """
    n_members = int(fixed_pdf.info["NumMembers"])

    if fixed_pdf_member == "replica":
        if fixed_pdf.error_type != "replicas":
            raise ValueError(
                f"fixed_pdf_member: replica needs a Monte Carlo PDF set, but "
                f"{fixed_pdf.name} has error type '{fixed_pdf.error_type}'."
            )
        if replica_index is None:
            raise ValueError(
                "fixed_pdf_member: replica needs a replica index (run with -rep N)."
            )
        member = int(replica_index)
    else:
        member = int(fixed_pdf_member)

    if not 0 <= member < n_members:
        raise ValueError(
            f"Member {member} not available in {fixed_pdf.name}, which has "
            f"members 0 to {n_members - 1}."
        )

    log.info(f"Using fixed PDF {fixed_pdf.name}, member {member}, at Q0 = {Q0} GeV.")

    if positivity_penalty_settings and positivity_penalty_settings.get(
        "positivity_penalty"
    ):
        log.warning(
            "The PDF is fixed: the positivity penalty is a constant and has no effect "
            "on the fit. Consider setting positivity_penalty: false."
        )
    if integrability_settings is not None and integrability_settings.integrability:
        log.warning(
            "The PDF is fixed: the integrability penalty is a constant and has no "
            "effect on the fit. Consider setting integrability: false."
        )

    model = FixedPDFModel(fixed_pdf.name, Q0, member)

    if dump_model:
        import dill

        with open(output_path / "pdf_model.pkl", "wb") as file:
            dill.dump(model, file)

    return model
