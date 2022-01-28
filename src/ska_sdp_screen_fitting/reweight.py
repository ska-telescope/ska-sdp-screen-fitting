"""
Reweight module
"""

import numpy as np

from ska_sdp_screen_fitting._logging import logger as logging
from ska_sdp_screen_fitting.lib_operations import (
    MultiprocManager,
    normalize_phase,
)

logging.debug("Loading REWEIGHT module.")


def _run_parser(soltab, parser, step):
    mode = parser.getstr(step, "mode", "uniform")
    weight_val = parser.getfloat(step, "weight_val", 1.0)
    nmedian = parser.getint(step, "nmedian", 3)
    nstddev = parser.getint(step, "nstddev", 251)
    soltab_import = parser.getstr(step, "soltab_import", "")
    flag_bad = parser.getbool(step, "flag_bad", False)
    ncpu = parser.getint("_global", "ncpu", 0)

    parser.checkSpelling(
        step,
        soltab,
        [
            "mode",
            "weight_val",
            "nmedian",
            "nstddev",
            "soltab_import",
            "flag_bad",
        ],
    )
    return run(
        soltab,
        mode,
        weight_val,
        nmedian,
        nstddev,
        soltab_import,
        flag_bad,
        ncpu,
    )


def _rolling_window_lastaxis(array, window):
    """
    Directly taken from Erik Rigtorp's post to numpy-discussion.
    <http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html>

    Make an ndarray with a rolling window of the last dimension

    Parameters
    ----------
    array : array_like
        Array to add rolling window to
    window : int
        Size of rolling window

    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.

    """

    if window < 1:
        raise ValueError("`window` must be at least 1.")
    if window > array.shape[-1]:
        raise ValueError("`window` is too long.")
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)


def _nancircstd(samples, axis=None, is_phase=True):
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
        x_2 = np.sqrt(1.0 - x_1 ** 2)
    R = np.hypot(np.nanmean(x_1, axis=axis), np.nanmean(x_2, axis=axis))

    return np.sqrt(-2 * np.log(R))


def _estimate_weights_window(sindx, vals, nmedian, nstddev, stype, out_queue):
    """
    Set weights using a median-filter method

    Parameters
    ----------
    sindx: int
        Index of station
    vals: array
        Array of values
    nmedian: odd int
        Size of median time window
    nstddev: odd int
        Size of stddev time window
    stype: str
        Type of values (e.g., 'phase')

    """

    pad_width = [(0, 0)] * len(vals.shape)
    pad_width[-1] = (int((nmedian - 1) / 2), int((nmedian - 1) / 2))
    if stype in ("phase", "rotation"):
        # Median smooth and subtract to de-trend
        if nmedian > 0:
            # Convert to real/imag
            real = np.cos(vals)
            pad_real = np.pad(
                real, pad_width, "constant", constant_values=(np.nan,)
            )
            med_real = np.nanmedian(
                _rolling_window_lastaxis(pad_real, nmedian), axis=-1
            )
            real -= med_real
            real[real < -1.0] = -1.0
            real[real > 1.0] = 1.0

            imag = np.sin(vals)
            pad_imag = np.pad(
                imag, pad_width, "constant", constant_values=(np.nan,)
            )
            med_imag = np.nanmedian(
                _rolling_window_lastaxis(pad_imag, nmedian), axis=-1
            )
            imag -= med_imag
            imag[imag < -1.0] = -1.0
            imag[imag > 1.0] = 1.0

            # Calculate standard deviations
            pad_width[-1] = (int((nstddev - 1) / 2), int((nstddev - 1) / 2))
            pad_real = np.pad(
                real, pad_width, "constant", constant_values=(np.nan,)
            )
            stddev1 = _nancircstd(
                _rolling_window_lastaxis(pad_real, nstddev),
                axis=-1,
                is_phase=False,
            )
            pad_imag = np.pad(
                imag, pad_width, "constant", constant_values=(np.nan,)
            )
            stddev2 = _nancircstd(
                _rolling_window_lastaxis(pad_imag, nstddev),
                axis=-1,
                is_phase=False,
            )
            stddev = stddev1 + stddev2
        else:
            phase = normalize_phase(vals)

            # Calculate standard deviation
            pad_width[-1] = (int((nstddev - 1) / 2), int((nstddev - 1) / 2))
            pad_phase = np.pad(
                phase, pad_width, "constant", constant_values=(np.nan,)
            )
            stddev = _nancircstd(
                _rolling_window_lastaxis(pad_phase, nstddev), axis=-1
            )
    else:
        if stype == "amplitude":
            # Assume lognormal distribution for amplitudes
            vals = np.log(vals)

        # Median smooth and subtract to de-trend
        if nmedian > 0:
            pad_vals = np.pad(
                vals, pad_width, "constant", constant_values=(np.nan,)
            )
            med = np.nanmedian(
                _rolling_window_lastaxis(pad_vals, nmedian), axis=-1
            )
            vals -= med

        # Calculate standard deviation in larger window
        pad_width[-1] = (int((nstddev - 1) / 2), int((nstddev - 1) / 2))
        pad_vals = np.pad(
            vals, pad_width, "constant", constant_values=(np.nan,)
        )
        stddev = np.nanstd(
            _rolling_window_lastaxis(pad_vals, nstddev), axis=-1
        )

    # Check for periods where standard deviation is zero or NaN and replace
    # with min value to prevent inf in the weights. Also limit weights to
    # float16
    zero_scatter_ind = np.where(np.logical_or(np.isnan(stddev), stddev == 0.0))
    if len(zero_scatter_ind[0]) > 0:
        good_ind = np.where(~np.logical_or(np.isnan(stddev), stddev == 0.0))
        stddev[zero_scatter_ind] = np.min(stddev[good_ind])
    if nmedian > 0:
        fudge_factor = 2.0  # factor to compensate for smoothing
    else:
        fudge_factor = 1.0
    weight = 1.0 / np.square(stddev * fudge_factor)

    # Rescale to fit in float16
    float16max = 65504.0
    if np.max(weight) > float16max:
        weight *= float16max / np.max(weight)

    out_queue.put([sindx, weight])


def run(
    soltab,
    mode="uniform",
    weight_val=1.0,
    nmedian=3,
    nstddev=251,
    soltab_import="",
    flag_bad=False,
    ncpu=0,
):
    """
    Change the the weight values.

    Parameters
    ----------
    mode : str, optional
        One of 'uniform' (single value), 'window' (sliding window in time), or
        'copy' (copy from another table), by default 'uniform'.
    weight_val : float, optional
        Set weights to this values (0=flagged), by default 1.
    nmedian : odd int, optional
        Median window size in number of timeslots for 'window' mode.
        If nonzero, a median-smoothed version of the input values is
        subtracted to detrend them. If 0, no smoothing or subtraction is
        done, by default 3.
    nstddev : odd int, optional
        Standard deviation window size in number of timeslots for 'window'
        mode, by default 251.
    soltab_import : str, optional
        Name of a soltab. Copy weights from this soltab (must have same
        axes shape), by default none.
    flag_bad : bool, optional
        Re-apply flags to bad values (1 for amp, 0 for other tables),
        by default False.
    """

    logging.info("Reweighting soltab: " + soltab.name)

    if mode == "copy":
        if soltab_import == "":
            logging.error("In copy mode a soltab_import must be specified.")
            return 1
        solset = soltab.getSolset()
        soltab_i = solset.getSoltab(soltab_import)
        soltab_i.selection = soltab.selection

        weights, axes = soltab.getValues(weight=True)
        weights_i, axes_i = soltab_i.getValues(weight=True)
        if (
            list(axes.keys()) != list(axes_i.keys())
            or weights.shape != weights_i.shape
        ):
            logging.error(
                "Impossible to merge: two tables have with different axes"
                " values."
            )
            return 1
        weights_i[np.where(weights == 0)] = 0.0
        soltab.setValues(weights_i, weight=True)
        soltab.addHistory("WEIGHT imported from " + soltab_i.name + ".")

    elif mode == "uniform":
        soltab.addHistory("REWEIGHTED to " + str(weight_val) + ".")
        soltab.setValues(weight_val, weight=True)

    elif mode == "window":
        if nmedian != 0 and nmedian % 2 == 0:
            logging.error("nmedian must be odd")
            return 1
        if nstddev % 2 == 0:
            logging.error("nstddev must be odd")
            return 1

        tindx = soltab.axesNames.index("time")
        antindx = soltab.axesNames.index("ant")
        vals = soltab.val[:].swapaxes(antindx, 0)
        if tindx == 0:
            tindx = antindx
        mpm = MultiprocManager(ncpu, _estimate_weights_window)
        for sindx, sval in enumerate(vals):
            if np.all(sval == 0.0) or np.all(np.isnan(sval)):
                # skip reference station
                continue
            mpm.put(
                [
                    sindx,
                    sval.swapaxes(tindx - 1, -1),
                    nmedian,
                    nstddev,
                    soltab.getType(),
                ]
            )
        mpm.wait()
        weights = np.ones(vals.shape)
        for (sindx, w) in mpm.get():
            weights[sindx, :] = w.swapaxes(-1, tindx - 1)
        weights = weights.swapaxes(0, antindx)

        soltab.addHistory(
            "REWEIGHTED using sliding window with nmedian={0} "
            "and nstddev={1} timeslots".format(nmedian, nstddev)
        )
        soltab.setValues(weights, weight=True)

    if flag_bad:
        weights = soltab.getValues(weight=True, retAxesVals=False)
        vals = soltab.getValues(retAxesVals=False)
        if soltab.getType() == "amplitude":
            weights[np.where(vals == 1)] = 0
        else:
            weights[np.where(vals == 0)] = 0
        soltab.setValues(weights, weight=True)

    return 0
