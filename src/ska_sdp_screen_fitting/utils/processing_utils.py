"""
    This script contains useful mathematical operations

    Copyright (c) 2022, SKAO / Science Data Processor
    SPDX-License-Identifier: BSD-3-Clause
"""

import errno
import os
import pickle
import shutil
from math import modf

import lsmtool
import numpy as np
import psutil
from astropy.io import fits as pyfits
from PIL import Image, ImageDraw
from shapely.geometry import Point, Polygon
from shapely.prepared import prep


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


def nancircstd(samples, axis=None, is_phase=True):
    """
    Compute the circular standard deviation

    Based on scipy.stats.circstd

    Parameters
    ----------
    imag : array_like
        Input array.
    axis : int, optional
        Axis along which standard deviations are computed.  The default is
        to compute the standard deviation of the flattened array.
    is_phase : bool, optional
        If True, samples are assumed to be phases. If False, they are assumed
        to be either real or imaginary values

    Returns
    -------
    circstd : float
        Circular standard deviation.
    """

    if is_phase:
        x_1 = np.sin(samples)
        x_2 = np.cos(samples)
    else:
        x_1 = samples
        x_2 = np.sqrt(1.0 - x_1**2)
    r_val = np.hypot(np.nanmean(x_1, axis=axis), np.nanmean(x_2, axis=axis))

    return np.sqrt(-2 * np.log(r_val))


def read_vertices(filename):
    """
    Returns facet vertices stored in input file
    """
    with open(filename, "rb") as file:
        vertices = pickle.load(file)
    return vertices


def make_template_image(
    image_name,
    reference_ra_deg,
    reference_dec_deg,
    ximsize=512,
    yimsize=512,
    cellsize_deg=0.000417,
    freqs=None,
    times=None,
    antennas=None,
    aterm_type="tec",
    fill_val=0,
):
    """
    Make a blank FITS image and save it to disk

    Parameters
    ----------
    image_name : str
        Filename of output image
    reference_ra_deg : float, optional
        RA for center of output mask image
    reference_dec_deg : float, optional
        Dec for center of output mask image
    imsize : int, optional
        Size of output image
    cellsize_deg : float, optional
        Size of a pixel in degrees
    freqs : list
        Frequencies to use to construct extra axes (for IDG a-term images)
    times : list
        Times to use to construct extra axes (for IDG a-term images)
    antennas : list
        Antennas to use to construct extra axes (for IDG a-term images)
    aterm_type : str
        One of 'tec' or 'gain'
    fill_val : int
        Value with which to fill the data
    """
    if freqs is not None and times is not None and antennas is not None:
        nants = len(antennas)
        ntimes = len(times)
        nfreqs = len(freqs)
        if aterm_type == "tec":
            # TEC solutions
            # data is [RA, DEC, ANTENNA, FREQ, TIME].T
            shape_out = [ntimes, nfreqs, nants, yimsize, ximsize]
        else:
            # Gain solutions
            # data is [RA, DEC, MATRIX, ANTENNA, FREQ, TIME].T
            shape_out = [ntimes, nfreqs, nants, 4, yimsize, ximsize]
    else:
        # Normal FITS image
        # data is [STOKES, FREQ, DEC, RA]
        shape_out = [1, 1, yimsize, ximsize]
        nfreqs = 1
        freqs = [150e6]

    hdu = pyfits.PrimaryHDU(np.ones(shape_out, dtype=np.float32) * fill_val)
    hdulist = pyfits.HDUList([hdu])
    header = hdulist[0].header

    # Add RA, Dec info
    i = 1
    header[f"CRVAL{i}"] = reference_ra_deg
    header[f"CDELT{i}"] = -cellsize_deg
    header[f"CRPIX{i}"] = ximsize / 2.0
    header[f"CUNIT{i}"] = "deg"
    header[f"CTYPE{i}"] = "RA---SIN"
    i += 1
    header[f"CRVAL{i}"] = reference_dec_deg
    header[f"CDELT{i}"] = cellsize_deg
    header[f"CRPIX{i}"] = yimsize / 2.0
    header[f"CUNIT{i}"] = "deg"
    header[f"CTYPE{i}"] = "DEC--SIN"
    i += 1

    # Add STOKES info or ANTENNA (+MATRIX) info
    if antennas is None:
        # basic image
        header[f"CRVAL{i}"] = 1.0
        header[f"CDELT{i}"] = 1.0
        header[f"CRPIX{i}"] = 1.0
        header[f"CUNIT{i}"] = ""
        header[f"CTYPE{i}"] = "STOKES"
        i += 1
    else:
        if aterm_type == "gain":
            # gain aterm images: add MATRIX info
            header[f"CRVAL{i}"] = 0.0
            header[f"CDELT{i}"] = 1.0
            header[f"CRPIX{i}"] = 1.0
            header[f"CUNIT{i}"] = ""
            header[f"CTYPE{i}"] = "MATRIX"
            i += 1

        # dTEC or gain: add ANTENNA info
        header[f"CRVAL{i}"] = 0.0
        header[f"CDELT{i}"] = 1.0
        header[f"CRPIX{i}"] = 1.0
        header[f"CUNIT{i}"] = ""
        header[f"CTYPE{i}"] = "ANTENNA"
        i += 1

    # Add frequency info
    ref_freq = freqs[0]
    if nfreqs > 1:
        deltas = freqs[1:] - freqs[:-1]
        del_freq = np.min(deltas)
    else:
        del_freq = 1e8
    header["RESTFRQ"] = ref_freq
    header[f"CRVAL{i}"] = ref_freq
    header[f"CDELT{i}"] = del_freq
    header[f"CRPIX{i}"] = 1.0
    header[f"CUNIT{i}"] = "Hz"
    header[f"CTYPE{i}"] = "FREQ"
    i += 1

    # Add time info
    if times is not None:
        ref_time = times[0]
        if ntimes > 1:
            # Find CDELT as the smallest delta time, but ignore last delta, as
            # it may be smaller due to the number of time slots not being a
            # divisor of the solution interval
            deltas = times[1:] - times[:-1]
            if ntimes > 2:
                del_time = np.min(deltas[:-1])
            else:
                del_time = deltas[0]
        else:
            del_time = 1.0
        header[f"CRVAL{i}"] = ref_time
        header[f"CDELT{i}"] = del_time
        header[f"CRPIX{i}"] = 1.0
        header[f"CUNIT{i}"] = "s"
        header[f"CTYPE{i}"] = "TIME"
        i += 1

    # Add equinox
    header["EQUINOX"] = 2000.0

    # Add telescope
    header["TELESCOP"] = "LOFAR"

    hdulist[0].header = header
    hdulist.writeto(image_name, overwrite=True)
    hdulist.close()


def rasterize(verts, data, blank_value=0):
    """
    Rasterize a polygon into a data array

    Parameters
    ----------
    verts : list of (x, y) tuples
        List of input vertices of polygon to rasterize
    data : 2-D array
        Array into which rasterize polygon
    blank_value : int or float, optional
        Value to use for blanking regions outside the poly

    Returns
    -------
    data : 2-D array
        Array with rasterized polygon
    """
    poly = Polygon(verts)
    prepared_polygon = prep(poly)

    # Mask everything outside of the polygon plus its border (outline) with
    # zeros (inside polygon plus border are ones)
    mask = Image.new("L", (data.shape[0], data.shape[1]), 0)
    ImageDraw.Draw(mask).polygon(verts, outline=1, fill=1)
    data *= mask

    # Now check the border precisely
    mask = Image.new("L", (data.shape[0], data.shape[1]), 0)
    ImageDraw.Draw(mask).polygon(verts, outline=1, fill=0)
    masked_ind = np.where(np.array(mask).transpose())
    points = [Point(xm, ym) for xm, ym in zip(masked_ind[0], masked_ind[1])]
    outside_points = [v for v in points if prepared_polygon.disjoint(v)]
    for outside_point in outside_points:
        data[int(outside_point.y), int(outside_point.x)] = 0

    if blank_value != 0:
        data[data == 0] = blank_value

    return data


def string2bool(invar):
    """
    Converts a string to a bool

    Parameters
    ----------
    invar : str
        String to be converted

    Returns
    -------
    result : bool
        Converted bool
    """
    if invar is None:
        return None
    if isinstance(invar, bool):
        return invar
    if isinstance(invar, str):
        if "TRUE" in invar.upper() or invar == "1":
            return True
        if "FALSE" in invar.upper() or invar == "0":
            return False
        raise ValueError(
            'input2bool: Cannot convert string "' + invar + '" to boolean!'
        )
    if isinstance(invar, float, int):
        return bool(invar)
    raise TypeError("Unsupported data type:" + str(type(invar)))


def string2list(invar):
    """
    Converts a string to a list

    Parameters
    ----------
    invar : str
        String to be converted

    Returns
    -------
    result : list
        Converted list
    """
    if invar is None:
        return None
    str_list = None
    if isinstance(invar, str):
        if invar.startswith("[") and invar.endswith("]"):
            str_list = [f.strip(" '\"") for f in invar.strip("[]").split(",")]
        elif "," in invar:
            str_list = [f.strip(" '\"") for f in invar.split(",")]
        else:
            str_list = [invar.strip(" '\"")]
    elif isinstance(invar, list):
        str_list = [str(f).strip(" '\"") for f in invar]
    else:
        raise TypeError("Unsupported data type:" + str(type(invar)))
    return str_list


def _float_approx_equal(x_coord, y_coord, tol=None, rel=None):
    if tol is rel is None:
        raise TypeError(
            "cannot specify both absolute and relative errors are None"
        )
    tests = []
    if tol is not None:
        tests.append(tol)
    if rel is not None:
        tests.append(rel * abs(x_coord))
    assert tests
    return abs(x_coord - y_coord) <= max(tests)


def approx_equal(x_coord, y_coord, *args, **kwargs):
    """
    Return True if x_coord and y_coord are approximately equal,
    otherwise False

    If x_coord and y_coord are floats, return True if y_coord is within either
    absolute error tol or relative error rel of x_coord. You can disable either
    the absolute or relative check by passing None as tol or rel (but not both)

    Parameters
    ----------
    x_coord : float
        First value to be compared
    y_coord : float
        Second value to be compared
    """
    if not type(x_coord) is type(y_coord) is float:
        # Skip checking for __approx_equal__ in the common case of two floats.
        methodname = "__approx_equal__"
        # Allow the objects to specify what they consider "approximately
        # equal", giving precedence to x. If either object has the appropriate
        # method, we pass on any optional arguments untouched.
        for a_coord, b_coord in ((x_coord, y_coord), (y_coord, x_coord)):
            try:
                method = getattr(a_coord, methodname)
            except AttributeError:
                continue
            else:
                result = method(b_coord, *args, **kwargs)
                if result is NotImplemented:
                    continue
                return bool(result)
    # If we get here without returning, then neither x nor y knows how to do an
    # approximate equal comparison (or are both floats). Fall back to a numeric
    # comparison.
    return _float_approx_equal(x_coord, y_coord, *args, **kwargs)


def create_directory(dirname):
    """
    Recursively create a directory, without failing if it already exists

    Parameters
    ----------
    dirname : str
        Path of directory
    """
    try:
        if dirname:
            os.makedirs(dirname)
    except OSError as failure:
        if failure.errno != errno.EEXIST:
            raise failure


def delete_directory(dirname):
    """
    Recursively delete a directory tree, without failing if it does not exist

    Parameters
    ----------
    dirname : str
        Path of directory
    """
    try:
        shutil.rmtree(dirname)
    except OSError as error:
        if error.errno != errno.ENOENT:
            raise error


def ra2hhmmss(deg):
    """
    Convert RA coordinate (in degrees) to HH MM SS

    Parameters
    ----------
    deg : float
        The RA coordinate in degrees

    Returns
    -------
    hour : int
        The hour (HH) part
    minute : int
        The minute (MM) part
    second : float
        The second (SS) part
    """
    deg = deg % 360
    x_coord, hour = modf(deg / 15)
    x_coord, minute = modf(x_coord * 60)
    second = x_coord * 60

    return (int(hour), int(minute), second)


def dec2ddmmss(deg):
    """
    Convert Dec coordinate (in degrees) to DD MM SS

    Parameters
    ----------
    deg : float
        The Dec coordinate in degrees

    Returns
    -------
    degree : int
        The degree (DD) part
    arcmin : int
        The arcminute (MM) part
    arcsec : float
        The arcsecond (SS) part
    sign : int
        The sign (+/-)
    """
    sign = -1 if deg < 0 else 1
    x_coord, degree = modf(abs(deg))
    x_coord, arcmin = modf(x_coord * 60)
    arcsec = x_coord * 60

    return (int(degree), int(arcmin), arcsec, sign)


def get_reference_station(soltab, max_ind=None):
    """
    Return the index of the station with the lowest fraction of flagged
    solutions

    Parameters
    ----------
    soltab : losoto solution table object
        The input solution table
    max_ind : int, optional
        The maximum station index to use when choosing the reference
        station. The reference station will be drawn from the first
        max_ind stations. If None, all stations are considered.

    Returns
    -------
    ref_ind : int
        Index of the reference station
    """
    if max_ind is None or max_ind > len(soltab.ant):
        max_ind = len(soltab.ant)

    weights = soltab.get_values(ret_axes_vals=False, weight=True)
    weights = np.sum(
        weights,
        axis=tuple(
            [
                i
                for i, axis_name in enumerate(soltab.get_axes_names())
                if axis_name != "ant"
            ]
        ),
        dtype=np.float,
    )
    ref_ind = np.where(weights[0:max_ind] == np.max(weights[0:max_ind]))[0][0]

    return ref_ind


def remove_soltabs(solset, soltabnames):
    """
    Remove H5parm soltabs from a solset

    Note: the H5parm must be opened with readonly = False

    Parameters
    ----------
    solset : losoto solution set object
        The solution set from which to remove soltabs
    soltabnames : list
        Names of soltabs to remove
    """
    soltabnames = string2list(soltabnames)
    for soltabname in soltabnames:
        try:
            soltab = solset.getSoltab(soltabname)
            soltab.delete()
        except Exception:
            print(f'Error: soltab "{soltabname}" could not be removed')


def get_available_memory():
    """
    Returns the available memory in GB

    Note: a call to 'free' is used, which is parsed for the "available" value,
    the last entry on the second line of output.

    Returns
    -------
    available_gb : int
        Available memory in GB
    """

    mem = psutil.virtual_memory()
    available_gb = int(np.floor(mem.available / 1024 / 1024 / 1024))
    return available_gb


def read_patch_list(skymodel, h5_file, soltab):

    """Read patch coordinates from skymodel file
    Parameters
    ----------
    skymodel : string
        Path to skymodel file
    h5_file : string
        Path to h5 file
    soltab : string
        Solution tab name

    Returns
    -------
    source_positions : array
        Source position read from skymodel (ra-dec)
    """

    skymod = lsmtool.load(skymodel)
    source_dict = skymod.getPatchPositions()
    source_positions = []
    for source in list(h5_file[f"sol000/{soltab}/dir"]):
        radecpos = source_dict[str(source, "utf-8").strip("[]")]
        source_positions.append([radecpos[0].value, radecpos[1].value])
    source_positions = np.array(source_positions)
    return source_positions


def get_patch_coordinates(source_positions, wcs_obj):

    """Convert coordinates from ra-dec to pixel coordinates
    Parameters
    ----------
    source_positions : array
        Source position read from skymodel (ra-dec)
    wcs_obj : wcs object
        Wcs object containing input fits information for the conversion

    Returns
    -------
    coord_patch_x : array
        Pixel x-coordinate of sources
    coord_patch_y : array
        Pixel y-coordinate of sources
    """

    ra_ind = wcs_obj.axis_type_names.index("RA")
    dec_ind = wcs_obj.axis_type_names.index("DEC")

    # Convert ra-dec to pixel coordinates
    ras = source_positions[:, 0]
    dec = source_positions[:, 1]
    xy_coord = []
    coord_patch_x = []
    coord_patch_y = []
    for ra_vert, dec_vert in zip(ras, dec):
        ra_dec = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        ra_dec[0][ra_ind] = ra_vert
        ra_dec[0][dec_ind] = dec_vert
        xy_coord.append(
            (
                wcs_obj.wcs_world2pix(ra_dec, 0)[0][ra_ind],
                wcs_obj.wcs_world2pix(ra_dec, 0)[0][dec_ind],
            )
        )
        coord_patch_x.append(
            int(np.round(wcs_obj.wcs_world2pix(ra_dec, 0)[0][dec_ind]))
        )
        coord_patch_y.append(
            int(np.round(wcs_obj.wcs_world2pix(ra_dec, 0)[0][ra_ind]))
        )

    return coord_patch_x, coord_patch_y
