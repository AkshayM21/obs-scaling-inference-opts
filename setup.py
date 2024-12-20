from setuptools import setup, find_packages

setup(
    name="evaluate_test",
    packages=find_packages(),
    install_requires=[
        "accelerate",
        "pandas",
        "huggingface"
    ]
)
