from validphys.app import App
from super_net.config import SuperNetConfig


providers = [
    "reportengine.report",
    "super_net.theory_predictions",
    "super_net.loss_functions",
    "super_net.optax_optimizer",
    "super_net.data_batch",
    "super_net.monte_carlo_utils",
    "super_net.commondata_utils",
    "super_net.closure_test.closure_test_estimators",
    "super_net.training_validation",
    "super_net.plots_and_tables.plotting",
]


class SuperNetApp(App):
    config_class = SuperNetConfig


def main():
    a = SuperNetApp(name="super_net", providers=providers)
    a.main()


if __name__ == "__main__":
    main()
