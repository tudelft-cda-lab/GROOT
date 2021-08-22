import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="groot-trees",
    version="0.0.10",
    author="Daniel Vos",
    author_email="D.A.Vos@tudelft.nl",
    url="https://github.com/tudelft-cda-lab/GROOT",
    download_url="https://github.com/tudelft-cda-lab/GROOT/archive/refs/tags/v0.0.1.tar.gz",
    description="Growing Robust Decision Trees",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    py_modules=["groot"],
    install_requires=[
        "dill",
        "joblib",
        "matplotlib",
        "numba",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "tqdm",
    ]
)