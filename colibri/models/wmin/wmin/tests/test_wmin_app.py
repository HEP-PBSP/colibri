import wmin


def test_wmin():
    """
    Test the wmin module is correctly imported.
    """
    assert wmin.__spec__.name == "wmin"
    assert wmin.__spec__.loader is not None
    assert wmin.__spec__.origin is not None
    assert wmin.__spec__.submodule_search_locations is not None
    assert wmin.__version__ is not None
