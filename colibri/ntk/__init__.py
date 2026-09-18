"""
colibri.ntk

NTK (Neural Tangent Kernel) analysis module for colibri.

This module provides tools for computing and analyzing the Neural Tangent Kernel
of PDF fits, including eigenvalue/eigenvector computation and plotting utilities.
"""

from colibri.ntk.ntkutils import NTKGrid, NTKStats
from colibri.ntk.eigenvalues import (
    EigenvalueGrid,
    eigenvalue_grid,
    eigenvalues_ensemble,
)
from colibri.ntk.eigenvector import (
    EigenvectorGrid,
    eigenvector_grid,
    eigenvectors_ensemble_at_epoch,
)

__all__ = [
    # Abstract interface
    "NTKGrid",
    "NTKStats",
    # Eigenvalue classes and functions
    "EigenvalueGrid",
    "eigenvalue_grid",
    "eigenvalues_ensemble",
    # Eigenvector classes and functions
    "EigenvectorGrid",
    "eigenvector_grid",
    "eigenvectors_ensemble_at_epoch",
]
