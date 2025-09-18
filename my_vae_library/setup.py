# my_vae_library/setup.py
from setuptools import setup, find_packages

setup(
    name="my_vae_library",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "tensorflow_datasets",
    ],
)