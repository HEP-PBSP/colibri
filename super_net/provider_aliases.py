"""
super_net.provider_aliases.py

Module collecting aliases for functions when used as providers.

Author: Luca Mantani
Date: 21.12.2023
"""

###############################################
# Aliases of super_net.training_validation.py #
###############################################


def _data_values(make_data_values):
    """
    Internal alias function for make_data_values.
    """
    return make_data_values


def _posdata_split(make_posdata_split):
    """
    Internal alias function for make_posdata_split.
    """
    return make_posdata_split


##############################################
# Aliases of super_net.theory_predictions.py #
##############################################


def _pred_dataset(make_pred_dataset):
    """
    Internal alias function for make_pred_dataset.
    """
    return make_pred_dataset


def _pred_data(make_pred_data):
    """
    Internal alias function for make_pred_data.
    """
    return make_pred_data


def _pred_t0data(make_pred_t0data):
    """
    Internal alias function for make_pred_t0data.
    """
    return make_pred_t0data


def _pred_data_non_vectorized(make_pred_data_non_vectorized):
    """
    Internal alias function for make_pred_data_non_vectorized.
    """
    return make_pred_data_non_vectorized


def _penalty_posdataset(make_penalty_posdataset):
    """
    Internal alias function for make_penalty_posdataset.
    """
    return make_penalty_posdataset


def _penalty_posdata(make_penalty_posdata):
    """
    Internal alias function for make_penalty_posdata.
    """
    return make_penalty_posdata


##########################################
# Aliases of super_net.loss_functions.py #
##########################################


def _chi2_training_data(make_chi2_training_data):
    """
    Internal alias function for make_chi2_training_data.
    """
    return make_chi2_training_data


def _chi2_training_data_with_positivity(make_chi2_training_data_with_positivity):
    """
    Internal alias function for make_chi2_training_data_with_positivity.
    """
    return make_chi2_training_data_with_positivity


def _chi2_validation_data(make_chi2_validation_data):
    """
    Internal alias function for make_chi2_validation_data.
    """
    return make_chi2_validation_data


def _chi2_validation_data_with_positivity(make_chi2_validation_data_with_positivity):
    """
    Internal alias function for make_chi2_validation_data_with_positivity.
    """
    return make_chi2_validation_data_with_positivity


def _chi2(make_chi2):
    """
    Internal alias function for make_chi2.
    """
    return make_chi2


def _chi2_with_positivity(make_chi2_with_positivity):
    """
    Internal alias function for make_chi2_with_positivity.
    """
    return make_chi2_with_positivity
