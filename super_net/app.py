from validphys.app import App
from super_net.config import SuperNetConfig


providers = [
    "reportengine.report",
    "super_net.loss_utils",
    "super_net.theory_predictions",
    "super_net.wmin_model",
    "super_net.loss_functions",
    "super_net.optax_optimizer",
    "super_net.data_batch",
]


class SuperNetApp(App):
    config_class = SuperNetConfig


def main():
    a = SuperNetApp(name="super_net", providers=providers)
    a.main()


if __name__ == "__main__":
    main()
