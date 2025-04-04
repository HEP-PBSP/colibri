"""
les_houches_example.app.py

"""

from colibri.app import colibriApp
from les_houches_example.config import ExampleConfig


ex_pdf_providers = [
    "les_houches_example.model",
]


class ExPdfApp(colibriApp):
    config_class = ExampleConfig


def main():
    a = ExPdfApp(name="gp", providers=ex_pdf_providers)
    a.main()


if __name__ == "__main__":
    main()
