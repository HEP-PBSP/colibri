"""
les_houches_example.app.py

"""

from colibri.app import colibriApp
from les_houches_example.config import LesHouchesConfig


lh_pdf_providers = [
    "les_houches_example.model",
]


class LesHouchesApp(colibriApp):
    config_class = LesHouchesConfig


def main():
    a = LesHouchesApp(name="gp", providers=lh_pdf_providers)
    a.main()


if __name__ == "__main__":
    main()
