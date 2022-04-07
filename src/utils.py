"""
auxillary functions
"""

import numpy as np


def get_absmax(ndarray):
    absmax = np.max(np.abs(ndarray))
    return absmax


def symmetric_clim(ndarray):
    """
    get 0-centered color limits (used when plotting with symmetric color maps)
    """
    absmax = get_absmax(ndarray)
    clim = (-absmax, absmax)
    return clim


def stack_ndarray_dict(dict_of_ndarrays):
    """
    converting a dict of equally-shaped ndarrays into a ndarray with one more dimension
    """
    if type(dict_of_ndarrays) == dict:
        ndarray = np.stack([*dict_of_ndarrays.values()])
        return ndarray
    else:
        print("Input is no dict, but " + str(type(dict_of_ndarrays)) + ". Returning object unchanged.")
        return dict_of_ndarrays
