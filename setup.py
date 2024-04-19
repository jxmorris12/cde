from setuptools import find_packages, setup

setup(
    name="spider",
    version="0.0.1",
    description="train semi-parametric text embedding models",
    author="Jack Morris",
    author_email="jxm3@cornell.edu",
    packages=find_packages(),
    install_requires=open("requirements.txt").readlines(),
)