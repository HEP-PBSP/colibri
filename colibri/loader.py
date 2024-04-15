"""
"""

import pathlib
import sys
import pkgutil
import os

from validphys.loader import LoaderBase, Loader
from reportengine.compat import yaml
from nnpdf_data import path_vpdata


def _get_colibri_profile():
    """
    Similar to the _get_nnpdf_profile function, this function returns the colibri profile as a dictionary.
    
    The profile yaml file is stored in pathlib.Path(sys.prefix) /share/COLIBRI/colibri_profile.yaml
    """
    yaml_reader = yaml.YAML(typ='safe', pure=True)

    profile_path = pathlib.Path(sys.prefix) / "share" / "COLIBRI" / "colibri_profile.yaml"
    import IPython; IPython.embed()
    # Set all default values
    profile_content = pkgutil.get_data("validphys", "nnprofile_default.yaml")
    profile_dict = yaml_reader.load(profile_content)
    # including the data_path to the validphys package
    profile_dict.setdefault("data_path", path_vpdata)

    # update profile dictionary
    with open(profile_path, encoding="utf-8") as f:
        profile_entries = yaml_reader.load(f)
        if profile_entries is not None:
            profile_dict.update(profile_entries)
    
    colibri_share = pathlib.Path(sys.prefix) / "share" / "COLIBRI"

    nnpdf_share = profile_dict.get("nnpdf_share")


    if nnpdf_share is None:
        if profile_path is not None:
            raise ValueError(
                f"`nnpdf_share` is not set in {profile_path}, please set it, e.g.: nnpdf_share: `.local/share/NNPDF`"
            )
        raise ValueError(
            "`nnpdf_share` not found in validphys, something is very wrong with the installation"
        )

    if nnpdf_share == "RELATIVE_TO_PYTHON":
        nnpdf_share = pathlib.Path(sys.prefix) / "share" / "NNPDF"

    # At this point nnpdf_share needs to be a path to somewhere
    nnpdf_share = pathlib.Path(nnpdf_share)

    # Make sure that we expand any ~ or ~<username>
    nnpdf_share = nnpdf_share.expanduser()

    # Make sure we can either write to this directory or it exists
    try:
        nnpdf_share.mkdir(exist_ok=True, parents=True)
    except PermissionError as e:
        raise FileNotFoundError(
            f"{nnpdf_share} does not exist and you haven't got permissions to create it!"
        ) from e

    # Now read all paths and define them as relative to nnpdf_share (unless given as absolute)
    for var in ["results_path", "theories_path", "validphys_cache_path", "hyperscan_path"]:
        # if there are any problems setting or getting these variable erroring out is more than justified
        absolute_var = nnpdf_share / pathlib.Path(profile_dict[var]).expanduser()
        profile_dict[var] = absolute_var.absolute().as_posix()

    return profile_dict


class ColibriLoaderBase(LoaderBase):
    """
    Base class for the COLIBRI loader.
    Inherits from the validphys LoaderBase class but makes use of a different profile.
    """

    def __init__(self, profile=None):
        profile = _get_colibri_profile()
        super().__init__(profile=profile)
        

class Loader(ColibriLoaderBase, Loader):
    """
    Loader class for COLIBRI.
    Inherits from the ColibriLoaderBase and the validphys Loader class.
    """
    pass


if __name__ == "__main__":
    _get_colibri_profile()