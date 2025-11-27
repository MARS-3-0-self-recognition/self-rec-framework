from setuptools import setup, find_packages

setup(
    name="self-rec-framework",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "inspect-ai>=0.3.0",
        "pyyaml>=6.0",
    ],
)