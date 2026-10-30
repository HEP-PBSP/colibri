"""
colibri.tests.test_evolve_fit

Tests for the evolve_fit script.
"""

import logging
import sys
from unittest.mock import patch

import pytest


@pytest.fixture
def evolve_fit_module():
    # The script resets the root logger at import: restore it afterwards
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    from colibri.scripts import evolve_fit

    yield evolve_fit
    root.handlers[:] = handlers
    root.setLevel(level)


def test_evolve_fit_without_exportgrids(evolve_fit_module, tmp_path):
    """A fit without exportgrids (e.g. fixed_pdf) has nothing to evolve."""
    fit_dir = tmp_path / "fixed_pdf_fit"
    (fit_dir / "replicas" / "replica_1").mkdir(parents=True)

    with patch.object(sys, "argv", ["evolve_fit", str(fit_dir)]), patch.object(
        evolve_fit_module, "evolven3fit_main"
    ) as mock_evolve:
        with pytest.raises(ValueError, match="No exportgrids found"):
            evolve_fit_module.main()

    mock_evolve.assert_not_called()
    assert not (fit_dir / "nnfit").exists()
