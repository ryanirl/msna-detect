from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="msna-detect",
    version="0.1.0",
    description="A deep learning framework for automated detection of bursts in Muscle Sympathetic Nerve Activity.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Ryan 'RyanIRL' Peters",
    author_email="RyanIRL@icloud.com",
    url="https://github.com/ryanirl/msna-detect",
    packages=find_packages(),  # Multi-file package
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=required,
    keywords="msna, nerve activity, burst detection, deep learning, neural network, signal processing",
    project_urls={
        "Bug Reports": "https://github.com/ryanirl/msna-detect/issues",
        "Source": "https://github.com/ryanirl/msna-detect",
    },
)