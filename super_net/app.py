from validphys.app import App
from super_net.config import SuperNetConfig


providers = [
    "reportengine.report",
    "super_net.tmp"
]

class SuperNetApp(App):
    config_class = SuperNetConfig

def main():
    a = SuperNetApp(name='super_net', providers=providers)
    a.main()

if __name__ == '__main__':
    main()