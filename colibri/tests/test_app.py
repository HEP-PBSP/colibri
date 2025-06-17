import colibri
from colibri.app import colibriApp
from colibri.tests.conftest import CONFIG_YML_PATH


def test_version():
    assert colibri.__version__ == "1.0.0"


def test_app_initialisation():
    app_instance = colibriApp(name="test_colibriApp", providers=["test_provider"])
    assert app_instance.name == "test_colibriApp"

    # test that providers are correctly appended
    assert app_instance.default_providers[-1] == "test_provider"


def test_argparser():
    app_instance = colibriApp(name="test_colibriApp", providers=[])

    parser = app_instance.argparser

    args = parser.parse_args(
        [
            CONFIG_YML_PATH,
            "--replica_index",
            "1",
            "--trval_index",
            "2",
            "--output",
            "test_dir",
        ]
    )

    assert args.replica_index == 1
    assert args.trval_index == 2
    assert args.output == "test_dir"
