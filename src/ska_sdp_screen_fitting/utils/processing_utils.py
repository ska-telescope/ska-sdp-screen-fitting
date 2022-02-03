"""
    This script contains useful mathematical operations
    SPDX-License-Identifier: BSD-3-Clause
"""

import numpy as np


def reorder_axes(array, old_axes, new_axes):
    """
    Reorder axis of an array to match a new name pattern.

    Parameters
    ----------
    array : np array
        The array to transpose.
    old_axes : list of str
        A list like ['time','freq','pol'].
        It can contain more axes than the new list, those are ignored.
        This is to pass to oldAxis the soltab.getAxesNames() directly even on
        an array from getValuesIter()
    new_axes : list of str
        A list like ['time','pol','freq'].

    Returns
    -------
    np array
        With axis transposed to match the new_axes list.
    """
    old_axes = [ax for ax in old_axes if ax in new_axes]
    idx = [old_axes.index(ax) for ax in new_axes]
    return np.transpose(array, idx)


def remove_keys(dic, keys=[]):
    """
    Remove a list of keys from a dict and return a new one.

    Parameters
    ----------
    dic : dcit
        The input dictionary.
    keys : list of str
        A list of arguments to remove or a string for single argument.

    Returns
    -------
    dict
        Dictionary with removed keys.
    """
    dic_copy = dict(dic)
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        del dic_copy[key]
    return dic_copy


def normalize_phase(phase):
    """
    Normalize phase to the range [-pi, pi].

    Parameters
    ----------
    phase : array of float
        Phase to normalize.

    Returns
    -------
    array of float
        Normalized phases.
    """

    # Convert to range [-2*pi, 2*pi].
    out = np.fmod(phase, 2.0 * np.pi)
    # Remove nans
    nans = np.isnan(out)
    np.putmask(out, nans, 0)
    # Convert to range [-pi, pi]
    out[out < -np.pi] += 2.0 * np.pi
    out[out > np.pi] -= 2.0 * np.pi
    # Put nans back
    np.putmask(out, nans, np.nan)
    return out
