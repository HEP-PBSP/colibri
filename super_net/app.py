from reportengine.app import App
from reportengine.report import Config

class SuperNetConfig(Config):
    def produce_example(self):
        return "example"

class SuperNetApp(App):
    config_class = SuperNetConfig

def main():
    a = SuperNetApp(name='super_net', default_providers=['reportengine.report'])
    a.main()

if __name__ == '__main__':
    main()
