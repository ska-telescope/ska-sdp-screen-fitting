# pylint: disable=C0302
"""
This is the station-screen operation for LoSoTo
"""


import numpy as np
from astropy.wcs import WCS
from scipy.linalg import pinv, svd

from ska_sdp_screen_fitting import reweight
from ska_sdp_screen_fitting._logging import logger as logging
from ska_sdp_screen_fitting.lib_operations import (
    MultiprocManager,
    normalize_phase,
)

logging.debug("Loading STATIONSCREEN module.")


def _run_parser(soltab, parser, step):
    out_soltab = parser.getstr(step, "outSoltab")
    order = parser.getint(step, "Order", 5)
    beta = parser.getfloat(step, "Beta", 5.0 / 3.0)
    niter = parser.getint(step, "niter", 2)
    nsigma = parser.getfloat(step, "nsigma", 5.0)
    ref_ant = parser.getint(step, "RefAnt", -1)
    scale_order = parser.getbool(step, "ScaleOrder", True)
    scale_dist = parser.getfloat(step, "scaleDist", 25000.0)
    min_order = parser.getint(step, "MinOrder", 5)
    adjust_order = parser.getbool(step, "AdjustOrder", True)
    ncpu = parser.getint(step, "ncpu", 0)

    parser.checkSpelling(
        step,
        soltab,
        [
            "out_soltab",
            "order",
            "beta",
            "niter",
            "nsigma",
            "ref_ant",
            "scale_order",
            "scale_dist",
            "min_order",
            "adjust_order",
        ],
    )
    return run(
        soltab,
        out_soltab,
        order,
        beta,
        niter,
        nsigma,
        ref_ant,
        scale_order,
        scale_dist,
        min_order,
        adjust_order,
        ncpu,
    )


def _calculate_piercepoints(station_positions, source_positions):
    """
    Returns array of piercepoint locations

    Parameters
    ----------
    station_positions : array
        Array of station positions
    source_positions : array
        Array of source positions

    Returns
    -------
    piercepoints : array
        Array of pierce points
    mid_ra : float
        Reference RA for WCS system (deg)
    mid_dec : float
        Reference Dec for WCS system (deg)

    """

    logging.info("Calculating SCREEN pierce-point locations...")
    n_sources = source_positions.shape[0]
    n_stations = station_positions.shape[0]
    n_piercepoints = n_stations * n_sources
    piercepoints = np.zeros((n_piercepoints, 3))

    xyz = np.zeros((n_sources, 3))
    ra_deg = source_positions.T[0] * 180.0 / np.pi
    dec_deg = source_positions.T[1] * 180.0 / np.pi
    xy_coord, mid_ra, mid_dec = _getxy(ra_deg, dec_deg)
    xyz[:, 0] = xy_coord[0]
    xyz[:, 1] = xy_coord[1]
    pp_idx = 0
    for i in range(n_sources):
        for _ in station_positions:
            piercepoints[pp_idx, :] = xyz[i]
            pp_idx += 1

    return piercepoints, mid_ra, mid_dec


def _get_ant_dist(ant_xyz, ref_xyz):
    """
    Returns distance between ant and ref in m

     Parameters
    ----------
    ant_xyz : array
        Array of station position
    ref_xyz : array
        Array of reference position

    Returns
    -------
    dist : float
        Distance between station and reference positions

    """

    return np.sqrt(
        (ref_xyz[0] - ant_xyz[0]) ** 2
        + (ref_xyz[1] - ant_xyz[1]) ** 2
        + (ref_xyz[2] - ant_xyz[2]) ** 2
    )


def _getxy(ra_list, dec_list, mid_ra=None, mid_dec=None):
    """
    Returns array of projected x and y values.

    Parameters
    ----------
    ra_list, dec_list : list
        Lists of RA and Dec in degrees
    mid_ra : float
        RA for WCS reference in degrees
    mid_dec : float
        Dec for WCS reference in degrees

    Returns
    -------
    x_coord, y_coord : numpy array, numpy array, float, float
        arrays of x and y values

    """

    if mid_ra is None or mid_dec is None:
        x_coord, y_coord = _radec2xy(ra_list, dec_list)

        # Refine x and y using midpoint
        if len(x_coord) > 1:
            xmid = min(x_coord) + (max(x_coord) - min(x_coord)) / 2.0
            ymid = min(y_coord) + (max(y_coord) - min(y_coord)) / 2.0
            xind = np.argsort(x_coord)
            yind = np.argsort(y_coord)
            try:
                midxind = np.where(np.array(x_coord)[xind] > xmid)[0][0]
                midyind = np.where(np.array(y_coord)[yind] > ymid)[0][0]
                mid_ra = ra_list[xind[midxind]]
                mid_dec = dec_list[yind[midyind]]
                x_coord, y_coord = _radec2xy(
                    ra_list, dec_list, mid_ra, mid_dec
                )
            except IndexError:
                mid_ra = ra_list[0]
                mid_dec = dec_list[0]
        else:
            mid_ra = ra_list[0]
            mid_dec = dec_list[0]

    x_coord, y_coord = _radec2xy(
        ra_list, dec_list, ref_ra=mid_ra, ref_dec=mid_dec
    )

    return np.array([x_coord, y_coord]), mid_ra, mid_dec


def _radec2xy(ra_list, dec_list, ref_ra=None, ref_dec=None):
    """
    Returns x, y for input RA, Dec.

    Note that the reference RA and Dec must be the same in calls to both
    _radec2xy() and _xy2radec() if matched pairs of (x, y) <=> (RA, Dec) are
    desired.

    Parameters
    ----------
    ra_list : list
        List of RA values in degrees
    dec_list : list
        List of Dec values in degrees
    ref_ra : float, optional
        Reference RA in degrees.
    ref_dec : float, optional
        Reference Dec in degrees

    Returns
    -------
    x_coords, y_coords : list, list
        Lists of x and y pixel values corresponding to the input RA and Dec
        values

    """

    x_coords = []
    y_coords = []
    if ref_ra is None:
        ref_ra = ra_list[0]
    if ref_dec is None:
        ref_dec = dec_list[0]

    # Make wcs object to handle transformation from RA and Dec to pixel coords.
    wcs_object = _make_wcs(ref_ra, ref_dec)

    for ra_deg, dec_deg in zip(ra_list, dec_list):
        ra_dec = np.array([[ra_deg, dec_deg]])
        x_coords.append(wcs_object.wcs_world2pix(ra_dec, 0)[0][0])
        y_coords.append(wcs_object.wcs_world2pix(ra_dec, 0)[0][1])

    return x_coords, y_coords


def _xy2radec(x_coords, y_coords, ref_ra=0.0, ref_dec=0.0):
    """
    Returns x, y for input RA, Dec.

    Note that the reference RA and Dec must be the same in calls to both
    _radec2xy() and _xy2radec() if matched pairs of (x, y) <=> (RA, Dec) are
    desired.

    Parameters
    ----------
    x_coords : list
        List of x values in pixels
    y_coords : list
        List of y values in pixels
    ref_ra : float, optional
        Reference RA in degrees
    ref_dec : float, optional
        Reference Dec in degrees

    Returns
    -------
    ra_list, dec_list : list, list
        Lists of RA and Dec values corresponding to the input x and y pixel
        values

    """

    ra_list = []
    dec_list = []

    # Make wcs object to handle transformation from RA and Dec to pixel coords.
    wcs_object = _make_wcs(ref_ra, ref_dec)

    for x_point, y_point in zip(x_coords, y_coords):
        x_y = np.array([[x_point, y_point]])
        ra_list.append(wcs_object.wcs_pix2world(x_y, 0)[0][0])
        dec_list.append(wcs_object.wcs_pix2world(x_y, 0)[0][1])

    return ra_list, dec_list


def _make_wcs(ref_ra, ref_dec):
    """
    Makes simple WCS object.

    Parameters
    ----------
    ref_ra : float
        Reference RA in degrees
    ref_dec : float
        Reference Dec in degrees

    Returns
    -------
    wcs_object : astropy.wcs.WCS object
        A simple TAN-projection WCS object for specified reference position

    """

    wcs_object = WCS(naxis=2)
    wcs_object.wcs.crpix = [1000, 1000]
    wcs_object.wcs.cdelt = np.array([-0.0005, 0.0005])
    wcs_object.wcs.crval = [ref_ra, ref_dec]
    wcs_object.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs_object.wcs.set_pv([(2, 1, 45.0)])

    return wcs_object


def _flag_outliers(weights, residual, nsigma, screen_type):
    """
    Flags outliers

    Parameters
    ----------
    weights : array
        Array of weights
    phase_residual : array
        Array of residual values from phase screen fitting (rad)
    nsigma : float
        Number of sigma above with outliers are clipped (= weight set to zero)
    screen_type : str
        Type of screen: 'phase', 'tec', or 'amplitude'

    Returns
    -------
    weights : array
        array of weights, with flagged times set to 0


    """

    # Find stddev of the screen
    flagged = np.where(weights == 0.0)
    nonflagged = np.where(weights > 0.0)
    if nonflagged[0].size == 0:
        return weights

    if screen_type == "phase":
        # Use circular stddev
        residual = normalize_phase(residual)
        residual_nan = residual.copy()
        residual_nan[flagged] = np.nan
        screen_stddev = reweight._nancircstd(residual_nan, axis=0)
    elif screen_type in ("tec", "amplitude"):
        # Use normal stddev
        screen_stddev = np.sqrt(
            np.average(
                residual[nonflagged] ** 2, weights=weights[nonflagged], axis=0
            )
        )

    # Compare residuals to stddev of the screen
    outlier_ind = np.where(np.abs(residual) > nsigma * screen_stddev)
    weights[outlier_ind] = 0.0

    return weights


def _circ_chi2(samples, weights):
    """
    Compute the circular chi^2

    Based on scipy.stats.circstd

    Parameters
    ----------
    samples : array_like
        Input array.
    weights : array_like
        Input array.

    Returns
    -------
    chi2 : float
        Circular chi^2.
    """

    unflagged = np.where(weights > 0.0)
    if unflagged[0].size == 0:
        return 0.0

    x_1 = np.sin(samples[unflagged])
    x_2 = np.cos(samples[unflagged])
    meanx1, sumw = np.average(
        x_1 ** 2, weights=weights[unflagged], returned=True
    )
    meanx2, sumw = np.average(
        x_2 ** 2, weights=weights[unflagged], returned=True
    )
    r_val = np.hypot(meanx1, meanx2)
    var = 1.0 - r_val

    return var * sumw


def _calculate_svd(pierce_points, r_0, beta, n_piercepoints):
    """
    Returns result (unit_matrix) of svd for K-L vectors

    Parameters
    ----------
    pierce_points : array
        Array of piercepoint locations
    r_0: float
        Scale size of amp fluctuations (m)
    beta: float
        Power-law index for amp structure function (5/3 => pure Kolmogorov
        turbulence)
    n_piercepoints : int
        Number of piercepoints

    Returns
    -------
    c_matrix : array
        c_matrix matrix
    pinv_c : array
        Inv(c_matrix) matrix
    unit_matrix : array
        Unitary matrix

    """

    pierce_points_resized = np.resize(
        pierce_points, (n_piercepoints, n_piercepoints, 3)
    )
    pierce_points_resized = (
        np.transpose(pierce_points_resized, (1, 0, 2)) - pierce_points_resized
    )
    pierce_points_resized_squared = np.sum(pierce_points_resized ** 2, axis=2)
    c_matrix = (
        -((pierce_points_resized_squared / r_0 ** 2) ** (beta / 2.0)) / 2.0
    )
    pinv_c = pinv(c_matrix, rcond=1e-3)
    unit_matrix, _, _ = svd(c_matrix)

    return c_matrix, pinv_c, unit_matrix


def _fit_screen(
    source_names,
    full_matrices,
    pierce_points,
    array_to_fit,
    weights,
    order,
    r_0,
    beta,
    screen_type,
):
    """
    Fits a screen to amplitudes or phases using Karhunen-Lo`eve base vectors

    Parameters
    ----------
    source_names: array
        Array of source names
    full_matrices : list of arrays
        List of [c_matrix, pivC, unit_matrix] matrices for all piercepoints
    pierce_points: array
        Array of piercepoint locations
    airmass: array
        Array of airmass values (note: not currently used)
    array_to_fit: array
        Array of amp values to fit screen to
    weights: array
        Array of weights
    order: int
        Order of screen (i.e., number of KL base vectors to keep)
    r_0: float
        Scale size of amp fluctuations (m)
    beta: float
        Power-law index for amp structure function (5/3 => pure Kolmogorov
        turbulence)
    screen_type : str
        Type of screen: 'phase' or 'amplitude'

    Returns
    -------
    screen_fit_white_all, screen_residual_all : array, array
        Arrays of screen and residual (actual - screen) values

    """

    # Identify flagged directions
    n_sources_all = len(source_names)
    unflagged = np.where(weights > 0.0)
    n_sources = len(source_names[unflagged])

    # Initialize arrays
    n_piercepoints = n_sources
    n_piercepoints_all = n_sources_all
    screen_fit_all = np.zeros((n_sources_all, 1))
    pp_all = pierce_points.copy()
    rr_all = array_to_fit.copy()
    pierce_points = pp_all[unflagged[0], :]
    weights_unflagged = np.diag(weights[unflagged])

    # Calculate matrices
    if n_sources == n_sources_all:
        c_matrix, pinv_c, unit_matrix = FULL_MATRICES
    else:
        # Recalculate for unflagged directions
        c_matrix, pinv_c, unit_matrix = _calculate_svd(
            pierce_points, r_0, beta, n_piercepoints
        )

    arg00 = np.transpose(unit_matrix[:, :order])
    arg01 = np.dot(weights_unflagged, unit_matrix)[:, :order]
    arg1 = np.dot(arg00, arg01)
    inv_u = pinv(arg1, rcond=1e-3)

    # Fit screen to unflagged directions
    if screen_type == "phase":
        # Change phase to real/imag
        rr_real = np.cos(array_to_fit[unflagged])
        rr_imag = np.sin(array_to_fit[unflagged])

        # Calculate real screen
        rr1 = np.dot(
            np.transpose(unit_matrix[:, :order]),
            np.dot(weights_unflagged, rr_real),
        )
        real_fit = np.dot(
            pinv_c, np.dot(unit_matrix[:, :order], np.dot(inv_u, rr1))
        )

        # Calculate imag screen
        rr1 = np.dot(
            np.transpose(unit_matrix[:, :order]),
            np.dot(weights_unflagged, rr_imag),
        )
        imag_fit = np.dot(
            pinv_c, np.dot(unit_matrix[:, :order], np.dot(inv_u, rr1))
        )

        # Calculate phase screen
        screen_fit = np.arctan2(
            np.dot(c_matrix, imag_fit), np.dot(c_matrix, real_fit)
        )
        screen_fit_white = np.dot(pinv_c, screen_fit)
    elif screen_type == "amplitude":
        # Calculate log(amp) screen
        array_to_fit = array_to_fit[unflagged]
        rr1 = np.dot(
            np.transpose(unit_matrix[:, :order]),
            np.dot(weights_unflagged, np.log10(array_to_fit)),
        )
        amp_fit_log = np.dot(
            pinv_c, np.dot(unit_matrix[:, :order], np.dot(inv_u, rr1))
        )

        # Calculate amp screen
        screen_fit = np.dot(c_matrix, amp_fit_log)
        screen_fit_white = np.dot(pinv_c, screen_fit)
    elif screen_type == "tec":
        # Calculate tec screen
        array_to_fit = array_to_fit[unflagged]
        rr1 = np.dot(
            np.transpose(unit_matrix[:, :order]),
            np.dot(weights_unflagged, array_to_fit),
        )
        tec_fit = np.dot(
            pinv_c, np.dot(unit_matrix[:, :order], np.dot(inv_u, rr1))
        )

        # Calculate amp screen
        screen_fit = np.dot(c_matrix, tec_fit)
        screen_fit_white = np.dot(pinv_c, screen_fit)

    # Calculate screen in all directions
    if n_sources != n_sources_all:
        screen_fit_all[unflagged[0], :] = screen_fit[:, np.newaxis]
        flagged = np.where(weights <= 0.0)
        for findx in flagged[0]:
            p_val = pp_all[findx, :]
            diff_squared = np.sum(np.square(pierce_points - p_val), axis=1)
            c_val = -((diff_squared / (r_0 ** 2)) ** (beta / 2.0)) / 2.0
            screen_fit_all[findx, :] = np.dot(c_val, screen_fit_white)
        c_matrix, pinv_c, unit_matrix = full_matrices
        screen_fit_white_all = np.dot(pinv_c, screen_fit_all)
        if screen_type == "amplitude":
            screen_residual_all = rr_all - 10 ** (
                screen_fit_all.reshape(n_piercepoints_all)
            )
        else:
            screen_residual_all = rr_all - screen_fit_all.reshape(
                n_piercepoints_all
            )
    else:
        screen_fit_white_all = screen_fit_white
        if screen_type == "amplitude":
            screen_residual_all = rr_all - 10 ** (
                np.dot(c_matrix, screen_fit_white)
            )
        else:
            screen_residual_all = rr_all - np.dot(c_matrix, screen_fit_white)
    screen_fit_white_all = screen_fit_white_all.reshape((n_sources_all, 1))
    screen_residual_all = screen_residual_all.reshape((n_sources_all, 1))

    return (screen_fit_white_all, screen_residual_all)


def _process_station(
    array_to_fit,
    pierce_points,
    screen_order,
    station_weights,
    screen_type,
    niter,
    nsigma,
    adjust_order,
    source_names,
    full_matrix,
    beta,
    r_0,
):
    """
    Processes a station

    Parameters
    ----------
    soltab: solution table
        Soltab containing amplitude solutions
    outsoltab: str
        Name of output soltab
    order : int, optional
        Order of screen (i.e., number of KL base vectors to keep). If the order
        is scaled by dist (scale_order = True), the order is calculated as
        order * sqrt(dist/scale_dist)
    beta: float, optional
        Power-law index for amp structure function (5/3 => pure Kolmogorov
        turbulence)
    niter: int, optional
        Number of iterations to do when determining weights
    nsigma: float, optional
        Number of sigma above which directions are flagged
    ref_ant: str or int, optional
        Index (if int) or name (if str) of reference station (-1 => no ref)
    scale_order : bool, optional
        If True, scale the screen order with sqrt of distance/scale_dist to the
        reference station
    scale_dist : float, optional
        Distance used to normalize the distances used to scale the screen
        order. If None, the max distance is used
    adjust_order : bool, optional
        If True, adjust the screen order to obtain a reduced chi^2 of approx.
        unity
    min_order : int, optional
        The minimum allowed order if adjust_order = True.

    """
    # Iterate:
    # 1. fit screens
    # 2. flag nsigma outliers
    # 3. refit with new weights
    # 4. repeat for niter
    target_redchi2 = 1.0
    n_sources = array_to_fit.shape[0]
    n_times = array_to_fit.shape[1]
    screen = np.zeros((n_sources, n_times))
    residual = np.zeros((n_sources, n_times))
    station_order = screen_order[0]
    init_station_weights = station_weights.copy()  # preserve initial weights
    for iterindx in range(niter):
        if iterindx > 0:
            # Flag outliers
            if screen_type in ("phase", "tec"):
                # Use residuals
                screen_diff = residual.copy()
            elif screen_type == "amplitude":
                # Use log residuals
                screen_diff = np.log10(array_to_fit) - np.log10(
                    np.abs(array_to_fit - residual)
                )
            station_weights = _flag_outliers(
                init_station_weights, screen_diff, nsigma, screen_type
            )

        prev_station_weights = init_station_weights
        prev_redchi2 = 0

        # Fit the screens
        norderiter = 1
        if adjust_order:
            if iterindx > 0:
                norderiter = 4
        for tindx in range(n_times):
            n_unflagged = np.where(station_weights[:, tindx] > 0.0)[0].size
            if n_unflagged == 0:
                continue
            if screen_order[tindx] > n_unflagged - 1:
                screen_order[tindx] = n_unflagged - 1
            hit_upper = False
            hit_lower = False
            hit_upper2 = False
            hit_lower2 = False
            sign = 1.0
            for oindx in range(norderiter):
                skip_fit = False
                if iterindx > 0:
                    if np.all(
                        station_weights[:, tindx]
                        == prev_station_weights[:, tindx]
                    ):
                        if not adjust_order:
                            # stop fitting if weights did not change
                            break
                        if oindx == 0:
                            # Skip the fit for first iteration, as it is the
                            # same as the prev one
                            skip_fit = True
                if (
                    not np.all(station_weights[:, tindx] == 0.0)
                    and not skip_fit
                ):
                    scr, res = _fit_screen(
                        source_names,
                        full_matrix,
                        pierce_points[:, :],
                        array_to_fit[:, tindx],
                        station_weights[:, tindx],
                        int(screen_order[tindx]),
                        r_0,
                        beta,
                        screen_type,
                    )
                    screen[:, tindx] = scr[:, 0]
                    residual[:, tindx] = res[:, 0]

                if hit_lower2 or hit_upper2:
                    break

                if adjust_order and iterindx > 0:
                    if screen_type == "phase":
                        redchi2 = _circ_chi2(
                            residual[:, tindx], station_weights[:, tindx]
                        ) / (n_unflagged - screen_order[tindx])
                    elif screen_type == "amplitude":
                        # Use log residuals
                        screen_diff = np.log10(
                            array_to_fit[:, tindx]
                        ) - np.log10(
                            np.abs(array_to_fit[:, tindx] - residual[:, tindx])
                        )
                        redchi2 = np.sum(
                            np.square(screen_diff) * station_weights[:, tindx]
                        ) / (n_unflagged - screen_order[tindx])
                    else:
                        redchi2 = np.sum(
                            np.square(residual[:, tindx])
                            * station_weights[:, tindx]
                        ) / (n_unflagged - screen_order[tindx])
                    if oindx > 0:
                        if redchi2 > 1.0 and prev_redchi2 < redchi2:
                            sign *= -1
                        if redchi2 < 1.0 and prev_redchi2 > redchi2:
                            sign *= -1
                    prev_redchi2 = redchi2
                    order_factor = (n_unflagged - screen_order[tindx]) ** 0.2
                    target_order = float(
                        screen_order[tindx]
                    ) - sign * order_factor * (target_redchi2 - redchi2)
                    target_order = max(station_order, target_order)
                    target_order = min(
                        int(round(target_order)), n_unflagged - 1
                    )
                    if target_order <= 0:
                        target_order = min(station_order, n_unflagged - 1)
                    if (
                        target_order == screen_order[tindx]
                    ):  # don't fit again if order is the same as last one
                        break
                    if (
                        target_order == n_unflagged - 1
                    ):  # check whether we've been here before. If so, break
                        if hit_upper:
                            hit_upper2 = True
                        hit_upper = True
                    if (
                        target_order == station_order
                    ):  # check whether we've been here before. If so, break
                        if hit_lower:
                            hit_lower2 = True
                        hit_lower = True
                    screen_order[tindx] = target_order
        prev_station_weights = station_weights.copy()

    return screen, station_weights, residual, screen_order


def _process_single_freq(
    freq_ind,
    screen_type,
    niter,
    nsigma,
    ref_ant,
    adjust_order,
    source_names,
    station_names,
    beta,
    r_0,
    out_queue,
):
    global R_FULL, WEIGHTS_FULL, SCREEN, RESIDUAL, FULL_MATRICES, SCREEN_ORDER
    global PIERCE_POINTS

    n_sources, _, n_times, _, n_pols = SCREEN.shape
    weights_out = np.zeros(WEIGHTS_FULL[:, :, 0, :, :].shape)
    screen_out = np.zeros(SCREEN[:, :, :, 0, :].shape)
    residual_out = np.zeros(RESIDUAL[:, :, :, 0, :].shape)
    screen_order_out = np.zeros(SCREEN_ORDER[:, :, 0, :].shape)
    for pol_ind in range(n_pols):
        residual = R_FULL[
            :, :, freq_ind, :, pol_ind
        ]  # order is now [dir, time, ant]
        residual = residual.transpose(
            [0, 2, 1]
        )  # order is now [dir, ant, time]
        weights = WEIGHTS_FULL[:, :, freq_ind, :, pol_ind]
        weights = weights.transpose([0, 2, 1])

        # Fit screens
        for station, _ in enumerate(station_names):
            if station == ref_ant and (screen_type in ("phase", "tec")):
                # skip reference station (phase- or tec-type only)
                continue
            if np.all(np.isnan(residual[:, station, :])) or np.all(
                weights[:, station, :] == 0
            ):
                # skip fully flagged stations
                continue
            array_to_fit = np.reshape(
                residual[:, station, :], [n_sources, n_times]
            )
            station_weights = weights[:, station, :]
            station_screen_order = SCREEN_ORDER[station, :, freq_ind, pol_ind]
            scr, wgt, res, sord = _process_station(
                array_to_fit,
                PIERCE_POINTS,
                station_screen_order,
                station_weights,
                screen_type,
                niter,
                nsigma,
                adjust_order,
                source_names,
                FULL_MATRICES,
                beta,
                r_0,
            )
            screen_out[:, station, :, pol_ind] = scr
            weights[:, station, :] = wgt
            residual_out[:, station, :, pol_ind] = res
            screen_order_out[station, :, pol_ind] = sord
        weights_out[:, :, :, pol_ind] = weights.transpose(
            [0, 2, 1]
        )  # order is now [dir, time, ant]

    out_queue.put(
        [freq_ind, screen_out, weights_out, residual_out, screen_order_out]
    )


def run(
    soltab,
    outsoltab,
    order=12,
    beta=5.0 / 3.0,
    niter=2,
    nsigma=5.0,
    ref_ant=-1,
    scale_order=True,
    scale_dist=None,
    min_order=5,
    adjust_order=True,
    ncpu=0,
):
    """
    Fits station screens to input soltab (type 'phase' or 'amplitude' only).

    The results of the fit are stored in the soltab parent solset in
    "outsoltab" and the RESIDUAL values (actual - screen) are stored in
    "outsoltabresid". These values are the screen amplitude values per station
    per pierce point per solution interval. The pierce point locations are
    stored in an auxiliary array in the output soltabs.

    Screens can be plotted with the PLOTSCREEN operation.

    Parameters
    ----------
    soltab: solution table
        Soltab containing amplitude solutions
    outsoltab: str
        Name of output soltab
    order : int, optional
        Order of screen (i.e., number of KL base vectors to keep). If the order
        is scaled by dist (scale_order = True), the order is calculated as
        order * sqrt(dist/scale_dist)
    beta: float, optional
        Power-law index for amp structure function (5/3 => pure Kolmogorov
        turbulence)
    niter: int, optional
        Number of iterations to do when determining weights
    nsigma: float, optional
        Number of sigma above which directions are flagged
    ref_ant: str or int, optional
        Index (if int) or name (if str) of reference station (-1 => no ref)
    scale_order : bool, optional
        If True, scale the screen order with sqrt of distance/scale_dist to the
        reference station
    scale_dist : float, optional
        Distance used to normalize the distances used to scale the screen
        order. If None, the max distance is used
    adjust_order : bool, optional
        If True, adjust the screen order to obtain a reduced chi^2 of approx.
        unity
    min_order : int, optional
        The minimum allowed order if adjust_order = True.
    ncpu : int, optional
        Number of CPUs to use. If 0, all are used

    """

    global R_FULL, WEIGHTS_FULL, SCREEN, RESIDUAL, FULL_MATRICES, SCREEN_ORDER
    global PIERCE_POINTS

    # Get screen type
    screen_type = soltab.get_type()
    if screen_type not in ["phase", "amplitude", "tec"]:
        logging.error(
            'Screens can only be fit to soltabs of type "phase", "tec", or'
            ' "amplitude".'
        )
        return 1
    logging.info(
        "Using solution table {0} to calculate {1} "  # pylint: disable=C0209
        "screens".format(soltab.name, screen_type)
    )

    # Load values, etc.
    R_FULL = np.array(soltab.val)
    WEIGHTS_FULL = soltab.weight[:]
    times = np.array(soltab.time)
    freqs = soltab.freq[:]
    axis_names = soltab.get_axes_names()
    freq_ind = axis_names.index("freq")
    dir_ind = axis_names.index("dir")
    time_ind = axis_names.index("time")
    ant_ind = axis_names.index("ant")
    if "pol" in axis_names:
        is_scalar = False
        pol_ind = axis_names.index("pol")
        n_pols = len(soltab.pol[:])
        R_FULL = R_FULL.transpose(
            [dir_ind, time_ind, freq_ind, ant_ind, pol_ind]
        )
        WEIGHTS_FULL = WEIGHTS_FULL.transpose(
            [dir_ind, time_ind, freq_ind, ant_ind, pol_ind]
        )
    else:
        is_scalar = True
        n_pols = 1
        R_FULL = R_FULL.transpose([dir_ind, time_ind, freq_ind, ant_ind])
        R_FULL = R_FULL[:, :, :, :, np.newaxis]
        WEIGHTS_FULL = WEIGHTS_FULL.transpose(
            [dir_ind, time_ind, freq_ind, ant_ind]
        )
        WEIGHTS_FULL = WEIGHTS_FULL[:, :, :, :, np.newaxis]

    # Collect station and source names and positions and times, making sure
    # that they are ordered correctly.
    solset = soltab.get_solset()
    source_names = soltab.dir[:]
    source_dict = solset.get_source()
    source_positions = []
    for source in source_names:
        source_positions.append(source_dict[source])
    station_names = soltab.ant[:]
    if not isinstance(station_names, list):
        station_names = station_names.tolist()
    station_dict = solset.get_ant()
    station_positions = []
    for station in station_names:
        station_positions.append(station_dict[station])
    n_sources = len(source_names)
    n_times = len(times)
    n_stations = len(station_names)
    n_freqs = len(freqs)
    n_piercepoints = n_sources

    # Set ref station and reference phases if needed
    if isinstance(ref_ant, str):
        if n_stations == 1:
            ref_ant = -1
        elif ref_ant in station_names:
            ref_ant = station_names.index(ref_ant)
        else:
            ref_ant = -1
    if ref_ant != -1 and screen_type == "phase" or screen_type == "tec":
        r_ref = R_FULL[:, :, :, ref_ant, :].copy()
        for i in range(len(station_names)):
            R_FULL[:, :, :, i, :] -= r_ref

    if scale_order:
        dist = []
        if ref_ant == -1:
            station_order = [order] * n_stations
        else:
            for station_idx in range(len(station_names)):
                dist.append(
                    _get_ant_dist(
                        station_positions[station_idx],
                        station_positions[ref_ant],
                    )
                )
            if scale_dist is None:
                scale_dist = max(dist)
            logging.info(
                "Using variable order"  # pylint: disable=C0209
                " (with max order  = {0} "
                "and scaling dist = {1} m)".format(order, scale_dist)
            )
            station_order = []
            for station_idx in range(len(station_names)):
                station_order.append(
                    max(
                        min_order,
                        min(
                            order,
                            int(
                                order * np.sqrt(dist[station_idx] / scale_dist)
                            ),
                        ),
                    )
                )
    else:
        station_order = [order] * len(station_names)
        logging.info(
            "Using order = {0}".format(order)  # pylint: disable=C0209
        )

    # Initialize various arrays and parameters
    SCREEN = np.zeros((n_sources, n_stations, n_times, n_freqs, n_pols))
    RESIDUAL = np.zeros((n_sources, n_stations, n_times, n_freqs, n_pols))
    SCREEN_ORDER = np.zeros((n_stations, n_times, n_freqs, n_pols))
    for station_idx in range(len(station_names)):
        for freq_ind in range(n_freqs):
            for pol_ind in range(n_pols):
                SCREEN_ORDER[
                    station_idx, :, freq_ind, pol_ind
                ] = station_order[station_idx]
    r_0 = 100

    # Calculate full piercepoint arrays # maybe outdated, height not really
    # used .. ?
    # coordinate conversion RA-DEC to xy_coord (image plane)
    PIERCE_POINTS, mid_ra, mid_dec = _calculate_piercepoints(
        np.array([station_positions[0]]), np.array(source_positions)
    )
    FULL_MATRICES = _calculate_svd(PIERCE_POINTS, r_0, beta, n_piercepoints)

    # Fit station screens
    mpm = MultiprocManager(ncpu, _process_single_freq)
    for freq_ind in range(n_freqs):
        mpm.put(
            [
                freq_ind,
                screen_type,
                niter,
                nsigma,
                ref_ant,
                adjust_order,
                source_names,
                station_names,
                beta,
                r_0,
            ]
        )
    mpm.wait()
    for (freq_ind, scr, wgt, res, sord) in mpm.get():
        SCREEN[:, :, :, freq_ind, :] = scr
        RESIDUAL[:, :, :, freq_ind, :] = res
        WEIGHTS_FULL[:, :, freq_ind, :, :] = wgt
        SCREEN_ORDER[:, :, freq_ind, :] = sord

    # Write the results to the output solset
    dirs_out = source_names
    times_out = times
    ants_out = station_names
    freqs_out = freqs

    # Store screen values
    # Note: amplitude screens are log10() values with non-log residuals!
    vals = SCREEN.transpose(
        [2, 3, 1, 0, 4]
    )  # order is now ['time', 'freq', 'ant', 'dir', 'pol']
    weights = WEIGHTS_FULL.transpose(
        [1, 2, 3, 0, 4]
    )  # order is now ['time', 'freq', 'ant', 'dir', 'pol']
    if is_scalar:
        screen_st = solset.make_soltab(
            f"{screen_type}screen",
            outsoltab,
            axes_names=["time", "freq", "ant", "dir"],
            axes_vals=[times_out, freqs_out, ants_out, dirs_out],
            vals=vals[:, :, :, :, 0],
            weights=weights[:, :, :, :, 0],
        )
        vals = RESIDUAL.transpose([2, 3, 1, 0, 4])
        weights = np.zeros(vals.shape)
        for source_idx in range(n_sources):
            # Store the screen order as the weights of the RESIDUAL soltab
            weights[:, :, :, source_idx, :] = SCREEN_ORDER.transpose(
                [1, 2, 0, 3]
            )  # order is now [time, ant, freq, pol]
        resscreen_st = solset.make_soltab(
            f"{screen_type}screenresid",
            outsoltab + "resid",
            axes_names=["time", "freq", "ant", "dir"],
            axes_vals=[times_out, freqs_out, ants_out, dirs_out],
            vals=vals[:, :, :, :, 0],
            weights=weights[:, :, :, :, 0],
        )
    else:
        pols_out = soltab.pol[:]
        screen_st = solset.make_soltab(
            f"{screen_type}screen",
            outsoltab,
            axes_names=["time", "freq", "ant", "dir", "pol"],
            axes_vals=[times_out, freqs_out, ants_out, dirs_out, pols_out],
            vals=vals,
            weights=weights,
        )
        vals = RESIDUAL.transpose([2, 3, 1, 0, 4])
        weights = np.zeros(vals.shape)
        for source_idx in range(n_sources):
            # Store the screen order as the weights of the RESIDUAL soltab
            weights[:, :, :, source_idx, :] = SCREEN_ORDER.transpose(
                [1, 2, 0, 3]
            )  # order is now [time, ant, freq, pol]
        resscreen_st = solset.make_soltab(
            f"{screen_type}screenresid",
            outsoltab + "resid",
            axes_names=["time", "freq", "ant", "dir", "pol"],
            axes_vals=[times_out, freqs_out, ants_out, dirs_out, pols_out],
            vals=vals,
            weights=weights,
        )

    # Store beta, r_0, height, and order as attributes of the screen soltabs
    screen_st.obj._v_attrs["beta"] = beta
    screen_st.obj._v_attrs["r_0"] = r_0
    screen_st.obj._v_attrs["height"] = 0.0
    screen_st.obj._v_attrs["midra"] = mid_ra
    screen_st.obj._v_attrs["middec"] = mid_dec

    # Store piercepoint table. Note that it does not conform to the axis
    # shapes, so we cannot use make_soltab()
    solset.obj._v_file.create_array(
        "/" + solset.name + "/" + screen_st.obj._v_name,
        "piercepoint",
        obj=PIERCE_POINTS,
    )

    screen_st.add_history("CREATE (by STATIONSCREEN operation)")
    resscreen_st.add_history("CREATE (by STATIONSCREEN operation)")

    return 0
