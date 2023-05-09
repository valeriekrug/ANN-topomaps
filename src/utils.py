import numpy as np


def get_absmax(ndarray):
    absmax = np.max(np.abs(ndarray))
    return absmax


def symmetric_clim(ndarray):
    absmax = get_absmax(ndarray)
    clim = (-absmax, absmax)
    return clim


def stack_ndarray_dict(dict_of_ndarrays):
    if type(dict_of_ndarrays) == dict:
        ndarray = np.stack([*dict_of_ndarrays.values()])
        return ndarray
    else:
        print("Input is no dict, but " + str(type(dict_of_ndarrays)) + ". Returning object unchanged.")
        return dict_of_ndarrays
