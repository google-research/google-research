"""setup.py for screen2words."""

from setuptools import find_packages
from setuptools import setup

# cd ..
# pip install -r rouge/requirements.txt
# pip install rouge-score

setup(
    name="screen2words",
    version="0.1",
    packages=find_packages(),
    author="Bryan Wang",
    author_email="bryanw@dgp.toronto.edu",
    install_requires=[
        "rouge-score>=0.1.2",
        "nltk>=3.8.1",
        "keras>=2.12.0",
        "tensorflow>=2.12.0",
        "tensorflow-intel>=2.12.0",
        "tf-models-official>=2.12.0",
        "tensorflow-text>=2.12.1",
        "tensorflow-metadata>=1.13.1"
    ],
)
