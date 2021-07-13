import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="groot-trees",                     # This is the name of the package
    version="0.0.4",                        # The initial release version
    author="Daniel Vos",                    # Full name of the author
    author_email="D.A.Vos@tudelft.nl",
    url="https://github.com/tudelft-cda-lab/GROOT",
    download_url="https://github.com/tudelft-cda-lab/GROOT/archive/refs/tags/v0.0.1.tar.gz",
    description="Growing Robust Decision Trees",
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),    # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.6',                # Minimum version requirement of the package
    py_modules=["groot"],                   # Name of the python package
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