import numpy as np


def convert_numpy(obj):
    """
    Convert numpy ints and floats to python types. Useful when converting objects to JSON.

    Parameters
    ----------
    obj : {np.int32, np.int64, np.float32, np.float64, np.longlong}
        Number to convert to python int or float.
    """
    if (
        isinstance(obj, np.int32)
        or isinstance(obj, np.int64)
        or isinstance(obj, np.longlong)
    ):
        return int(obj)
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    raise TypeError(f"Cannot convert type {type(obj)} to int or float")


def numpy_to_chensvmlight(X, y, filename):
    """
    Export a numpy dataset to the SVM-Light format that is needed for Chen et al. (2019).

    The difference between SVM-Light and this format is that zero values are also included.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Array of amples.
    y : array-like of shape (n_samples,)
        Array of class labels as integers.
    filename : str
        Exported SVM-Light dataset filename or path.
    """
    lines = []
    for sample, label in zip(X, y):
        terms = [str(label)]
        for i, value in enumerate(sample):
            terms.append(f"{i}:{value}")
        lines.append(" ".join(terms))
    svmlight_string = "\n".join(lines)

    with open(filename, "w") as file:
        file.write(svmlight_string)
