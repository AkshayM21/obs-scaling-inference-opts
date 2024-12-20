from setuptools import setup, find_packages


setup(
    name="evaluate_vllm",  # Changed to match directory name
    version="0.1.0",    # Added version
    packages=find_packages(),
    install_requires=[
        "accelerate",
        "pandas",
        "huggingface_hub",  # Corrected from "huggingface"
        # Removed "json" as it's part of Python's standard library
    ],
    entry_points={
        'console_scripts': [
            'evaluate_vllm=evaluate_vllm.__main__:main',  # Add this if you want command-line usage
        ],
    }
)
