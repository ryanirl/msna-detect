from setuptools import setup, find_packages

setup(
    name="msna-detect",
    version="0.1.3",
    description="A deep learning framework for automated detection of bursts in Muscle Sympathetic Nerve Activity.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Ryan 'RyanIRL' Peters",
    author_email="RyanIRL@icloud.com",
    url="https://github.com/ryanirl/msna-detect",
    packages=["msna_detect", "msna_detect.models", "msna_detect.utils", "scripts"],  # Multi-file package
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
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.10.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "tqdm>=4.60.0",
        "scikit-learn>=1.0.0",
        "bokeh>=2.4.0",
        "tornado>=6.1.0",
        "gdown==5.2.0"
    ],
    entry_points={
        "console_scripts": [
            "msna-detect-train=scripts.train:main",
            "msna-detect-predict=scripts.predict:main", 
            "msna-detect-eval=scripts.eval:main",
            "msna-detect-dashboard=scripts.dashboard:main",
        ],
    },
    keywords="msna, nerve activity, burst detection, deep learning, neural network, signal processing",
    project_urls={
        "Bug Reports": "https://github.com/ryanirl/msna-detect/issues",
        "Source": "https://github.com/ryanirl/msna-detect",
    },
)