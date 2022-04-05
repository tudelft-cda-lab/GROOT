import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="groot-trees",
    version="0.0.16",
    author="Daniel Vos",
    author_email="D.A.Vos@tudelft.nl",
    url="https://github.com/tudelft-cda-lab/GROOT",
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
        "wheel",
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