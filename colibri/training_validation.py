"""
colibri.training_validation.py

Module containing training validation dataclasses.

Author: Mark N. Costantini
Date: 11.11.2023
"""

import jax.numpy as jnp
from dataclasses import dataclass

from reportengine import collect
from reportengine.configparser import ConfigError

from colibri.commondata_utils import CentralCovmatIndex
from colibri.utils import training_validation_split, TrainValidationSplit


@dataclass(frozen=True)
class TrainCentralCovmatIndex(CentralCovmatIndex):
    n_training_points: int


@dataclass(frozen=True)
class ValidationCentralCovmatIndex(CentralCovmatIndex):
    n_validation_points: int


@dataclass(frozen=True)
class MakeDataValues:
    training_data: TrainCentralCovmatIndex = None
    validation_data: ValidationCentralCovmatIndex = None


@dataclass(frozen=True)
class PosdataTrainValidationSplit(TrainValidationSplit):
    n_training: int
    n_validation: int
