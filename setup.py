from setuptools import setup, find_packages

setup(
    name="evaluate",
    packages=find_packages(),
    install_requires=[
        "accelerate",
        "pandas",
        "json",
        "huggingface"
    ]
)