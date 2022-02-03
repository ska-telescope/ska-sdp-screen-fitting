"""
    Class and helper functions for KL (Karhunen-Lo`eve) screens
    
    Copyright (c) 2022, SKAO / Science Data Processor
    SPDX-License-Identifier: BSD-3-Clause
"""

import itertools
import multiprocessing
from multiprocessing import Pool, RawArray

import lsmtool
import numpy as np
from astropy import wcs

import ska_sdp_screen_fitting.utils.processing_utils as misc
from ska_sdp_screen_fitting import stationscreen
from ska_sdp_screen_fitting.screen import Screen
from ska_sdp_screen_fitting.utils.h5parm import H5parm


class KLScreen(Screen):
    """
    Class for KL (Karhunen-Lo`eve) screens
    """

    def __init__(
        self,
        name,
        h5parm_filename,
        skymodel_filename,
        rad,
        dec,
        width_ra,
        width_dec,
        solset_name="sol000",
        phase_soltab_name="phase000",
        amplitude_soltab_name=None,
    ):
        super(KLScreen, self).__init__(
            name,
            h5parm_filename,
            skymodel_filename,
            rad,
            dec,
            width_ra,
            width_dec,
            solset_name=solset_name,
            phase_soltab_name=phase_soltab_name,
            amplitude_soltab_name=amplitude_soltab_name,
        )

        # initialize extra members specific to KL
        self.height = None
        self.beta_val = None
        self.r_0 = None
        self.piercepoints = None
        self.mid_ra = None
        self.mid_dec = None

    def fit(self):
        """
        Fits screens to the input solutions
        """
        # Open solution tables
        h5_file = H5parm(self.input_h5parm_filename, readonly=False)
        solset = h5_file.get_solset(self.input_solset_name)
        soltab_ph = solset.get_soltab(self.input_phase_soltab_name)
        if not self.phase_only:
            soltab_amp = solset.get_soltab(self.input_amplitude_soltab_name)

        # Set the position of the calibration patches to those of
        # the input sky model, as the patch positions written to the H5parm
        # file by DPPP may be different
        skymod = lsmtool.load(self.input_skymodel_filename)
        source_dict = skymod.getPatchPositions()
        source_positions = []
        for source in soltab_ph.dir:
            radecpos = source_dict[source.strip("[]")]
            source_positions.append([radecpos[0].value, radecpos[1].value])
        source_positions = np.array(source_positions)
        # ra_deg = source_positions.T[0]
        # dec_deg = source_positions.T[1]
        # sourceTable = solset.obj._f_get_child("source")
        # vals = [
        #     [rad * np.pi / 180.0, dec * np.pi / 180.0]
        #     for rad, dec in zip(ra_deg, dec_deg)
        # ]
        # sourceTable = list(zip(*(soltab_ph.dir, vals)))

        # Now call the stationscreen operation to do the fitting. For the
        # phase screens, we reference the phases to the station with the least
        # amount of flagged solutions, drawn from the first 10 stations (to
        # ensure it is fairly central)
        ref_ind = misc.get_reference_station(soltab_ph, 10)
        adjust_order_amp = True
        screen_order_amp = min(
            12, max(3, int(np.round(len(source_positions) / 2)))
        )
        adjust_order_ph = True
        screen_order = min(20, len(source_positions) - 1)
        misc.remove_soltabs(solset, "phase_screen000")
        misc.remove_soltabs(solset, "phase_screen000resid")
        stationscreen.run(
            soltab_ph,
            "phase_screen000",
            order=screen_order,
            ref_ant=ref_ind,
            scale_order=True,
            adjust_order=adjust_order_ph,
            ncpu=self.ncpu,
        )
        soltab_ph_screen = solset.get_soltab("phase_screen000")
        if not self.phase_only:
            misc.remove_soltabs(solset, "amplitude_screen000")
            misc.remove_soltabs(solset, "amplitude_screen000resid")
            stationscreen.run(
                soltab_amp,
                "amplitude_screen000",
                order=screen_order_amp,
                niter=3,
                scale_order=False,
                adjust_order=adjust_order_amp,
                ncpu=self.ncpu,
            )
            soltab_amp_screen = solset.get_soltab("amplitude_screen000")
        else:
            soltab_amp_screen = None

        # Read in the screen solutions and parameters
        self.vals_ph = soltab_ph_screen.val
        self.times_ph = soltab_ph_screen.time
        self.freqs_ph = soltab_ph_screen.freq
        if not self.phase_only:
            self.log_amps = True
            self.vals_amp = soltab_amp_screen.val
            self.times_amp = soltab_amp_screen.time
            self.freqs_amp = soltab_amp_screen.freq
        self.source_names = soltab_ph_screen.dir
        self.source_dict = solset.get_source()
        self.source_positions = []
        for source in self.source_names:
            self.source_positions.append(self.source_dict[source])
        self.station_names = soltab_ph_screen.ant
        self.station_dict = solset.get_ant()
        self.station_positions = []
        for station in self.station_names:
            self.station_positions.append(self.station_dict[station])
        self.height = soltab_ph_screen.obj._v_attrs["height"]
        self.beta_val = soltab_ph_screen.obj._v_attrs["beta"]
        self.r_0 = soltab_ph_screen.obj._v_attrs["r_0"]
        self.piercepoints = np.array(soltab_ph_screen.obj.piercepoint)
        self.mid_ra = soltab_ph_screen.obj._v_attrs["midra"]
        self.mid_dec = soltab_ph_screen.obj._v_attrs["middec"]
        h5_file.close()

    def get_memory_usage(self, cellsize_deg):
        """
        Returns memory usage per time slot in GB

        Parameters
        ----------
        cellsize_deg : float
            Size of one pixel in degrees
        """
        ncpu = self.ncpu
        if ncpu == 0:
            ncpu = multiprocessing.cpu_count()

        # Make a test array and find its memory usage
        ximsize = int(self.width_ra / cellsize_deg)  # pix
        yimsize = int(self.width_dec / cellsize_deg)  # pix
        test_array = np.zeros(
            [
                1,
                len(self.freqs_ph),
                len(self.station_names),
                4,
                yimsize,
                ximsize,
            ]
        )
        mem_per_timeslot_gb = (
            test_array.nbytes / 1024**3 / 10
        )  # include factor of 10 overhead

        # Multiply by the number of CPUs, since each gets a copy
        mem_per_timeslot_gb *= ncpu

        return mem_per_timeslot_gb

    def make_matrix(
        self,
        t_start_index,
        t_stop_index,
        freq_ind,
        stat_ind,
        cellsize_deg,
        _,
        ncpu,
    ):
        """
        Makes the matrix of values for the given time, frequency, and station
        indices

        Parameters
        ----------
        t_start_index : int
            Index of first time
        t_stop_index : int
            Index of last time
        t_start_index : int
            Index of frequency
        t_stop_index : int
            Index of station
        cellsize_deg : float
            Size of one pixel in degrees
        out_dir : str
            Full path to the output directory
        ncpu : int, optional
            Number of CPUs to use (0 means all)
        """
        # Use global variables to avoid serializing the arrays in the
        # multiprocessing calls
        global SCREEN_PH, SCREEN_AMP_XX, SCREEN_AMP_YY, PIERCEPOINTS, X_COORD
        global Y_COORD, VAR_DICT

        # Define various parameters
        n_sources = len(self.source_names)
        n_times = t_stop_index - t_start_index
        n_piercepoints = n_sources
        beta_val = self.beta_val
        r_0 = self.r_0

        # Make arrays of pixel coordinates for screen
        # We need to convert the FITS cube pixel coords to screen pixel coords.
        # The FITS cube has self.rad, self.dec at (xsize/2, ysize/2)
        ximsize = int(np.ceil(self.width_ra / cellsize_deg))  # pix
        yimsize = int(np.ceil(self.width_dec / cellsize_deg))  # pix
        wcs_obj = wcs.WCS(naxis=2)
        wcs_obj.wcs.crpix = [ximsize / 2.0, yimsize / 2.0]
        wcs_obj.wcs.cdelt = np.array([-cellsize_deg, cellsize_deg])
        wcs_obj.wcs.crval = [self.rad, self.dec]
        wcs_obj.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs_obj.wcs.set_pv([(2, 1, 45.0)])

        x_fits = list(range(ximsize))
        y_fits = list(range(yimsize))
        rad = []
        dec = []
        for x_f, y_f in zip(x_fits, y_fits):
            x_y = np.array([[x_f, y_f]])
            rad.append(wcs_obj.wcs_pix2world(x_y, 0)[0][0])
            dec.append(wcs_obj.wcs_pix2world(x_y, 0)[0][1])
        x_y, _, _ = stationscreen._getxy(
            rad, dec, mid_ra=self.mid_ra, mid_dec=self.mid_dec
        )
        X_COORD = x_y[0].T
        Y_COORD = x_y[1].T
        len_x_coord = len(X_COORD)
        len_y_coord = len(Y_COORD)

        # Select input data and reorder the axes to get axis order of
        # [dir, time]
        # Input data are [time, freq, ant, dir, pol] for slow amplitudes
        # and [time, freq, ant, dir] for fast phases (scalarphase).
        time_axis = 0
        dir_axis = 1
        SCREEN_PH = np.array(
            self.vals_ph[t_start_index:t_stop_index, freq_ind, stat_ind, :]
        )
        SCREEN_PH = SCREEN_PH.transpose([dir_axis, time_axis])
        if not self.phase_only:
            SCREEN_AMP_XX = np.array(
                self.vals_amp[
                    t_start_index:t_stop_index, freq_ind, stat_ind, :, 0
                ]
            )
            SCREEN_AMP_XX = SCREEN_AMP_XX.transpose([dir_axis, time_axis])
            SCREEN_AMP_YY = np.array(
                self.vals_amp[
                    t_start_index:t_stop_index, freq_ind, stat_ind, :, 1
                ]
            )
            SCREEN_AMP_YY = SCREEN_AMP_YY.transpose([dir_axis, time_axis])

        # Process phase screens
        ncpu = self.ncpu
        if ncpu == 0:
            ncpu = multiprocessing.cpu_count()
        PIERCEPOINTS = self.piercepoints
        val_shape = (len_x_coord, len_y_coord, n_times)
        VAR_DICT = {}
        shared_val = RawArray(
            "d", int(val_shape[0] * val_shape[1] * val_shape[2])
        )
        screen_type = "ph"
        with Pool(
            processes=ncpu,
            initializer=init_worker,
            initargs=(shared_val, val_shape),
        ) as pool:
            pool.map(
                calculate_kl_screen_star,
                zip(
                    range(val_shape[2]),
                    itertools.repeat(n_piercepoints),
                    itertools.repeat(beta_val),
                    itertools.repeat(r_0),
                    itertools.repeat(screen_type),
                ),
            )
        val_phase = (
            np.frombuffer(shared_val, dtype=np.float64)
            .reshape(val_shape)
            .copy()
        )

        # Process amplitude screens
        if not self.phase_only:
            # XX amplitudes
            screen_type = "xx"
            with Pool(
                processes=ncpu,
                initializer=init_worker,
                initargs=(shared_val, val_shape),
            ) as pool:
                pool.map(
                    calculate_kl_screen_star,
                    zip(
                        range(val_shape[2]),
                        itertools.repeat(n_piercepoints),
                        itertools.repeat(beta_val),
                        itertools.repeat(r_0),
                        itertools.repeat(screen_type),
                    ),
                )
            val_amp_xx = 10 ** (
                np.frombuffer(shared_val, dtype=np.float64)
                .reshape(val_shape)
                .copy()
            )

            # YY amplitudes
            screen_type = "yy"
            with Pool(
                processes=ncpu,
                initializer=init_worker,
                initargs=(shared_val, val_shape),
            ) as pool:
                pool.map(
                    calculate_kl_screen_star,
                    zip(
                        range(val_shape[2]),
                        itertools.repeat(n_piercepoints),
                        itertools.repeat(beta_val),
                        itertools.repeat(r_0),
                        itertools.repeat(screen_type),
                    ),
                )
            val_amp_yy = 10 ** (
                np.frombuffer(shared_val, dtype=np.float64)
                .reshape(val_shape)
                .copy()
            )

        # Output data are [RA, DEC, MATRIX, ANTENNA, FREQ, TIME].T
        data = np.zeros((n_times, 4, len_y_coord, len_x_coord))
        if self.phase_only:
            data[:, 0, :, :] = np.cos(val_phase.T)
            data[:, 2, :, :] = np.cos(val_phase.T)
            data[:, 1, :, :] = np.sin(val_phase.T)
            data[:, 3, :, :] = np.sin(val_phase.T)
        else:
            data[:, 0, :, :] = val_amp_xx.T * np.cos(val_phase.T)
            data[:, 2, :, :] = val_amp_yy.T * np.cos(val_phase.T)
            data[:, 1, :, :] = val_amp_xx.T * np.sin(val_phase.T)
            data[:, 3, :, :] = val_amp_yy.T * np.sin(val_phase.T)

        return data


def init_worker(shared_val, val_shape):
    """
    Initializer called when a child process is initialized, responsible
    for storing store shared_val and val_shape in Var_dict (a global variable).

    See https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays
    -when-using-pythons-multiprocessing.html

    Parameters
    ----------
    shared_val : array
        RawArray to be shared
    val_shape : tuple
        Shape of shared_val array
    """
    global VAR_DICT

    VAR_DICT["shared_val"] = shared_val
    VAR_DICT["val_shape"] = val_shape


def calculate_kl_screen_star(inputs):
    """
    Simple helper function for pool.map
    """
    return calculate_kl_screen(*inputs)


def calculate_kl_screen(k, n_piercepoints, beta_val, r_0, screen_type):
    """
    Calculates screen images

    Parameters
    ----------
    k : int
        Time index
    n_piercepoints : int
        Number of pierce points
    beta_val : float
        Power-law index for phase structure function (5/3 =>
        pure Kolmogorov turbulence)
    r_0 : float
        Scale size of phase fluctuations
    screen_type : string
        Type of screen: 'ph'(phase), 'xx'(XX amplitude) or 'yy'(YY amplitude)
    """
    # Use global variables to avoid serializing the arrays in the
    # multiprocessing calls
    global SCREEN_PH, SCREEN_AMP_XX, SCREEN_AMP_YY, PIERCEPOINTS, X_COORD
    global Y_COORD, VAR_DICT

    tmp = np.frombuffer(VAR_DICT["shared_val"], dtype=np.float64).reshape(
        VAR_DICT["val_shape"]
    )
    if screen_type == "ph":
        inscreen = SCREEN_PH[:, k]
    if screen_type == "xx":
        inscreen = SCREEN_AMP_XX[:, k]
    if screen_type == "yy":
        inscreen = SCREEN_AMP_YY[:, k]
    f_matrix = inscreen.reshape(n_piercepoints)
    for i, x_i in enumerate(X_COORD):
        for j, y_i in enumerate(Y_COORD):
            p_array = np.array([x_i, y_i, 0.0])
            d_square = np.sum(np.square(PIERCEPOINTS - p_array), axis=1)
            c_matrix = -((d_square / (r_0**2)) ** (beta_val / 2.0)) / 2.0
            tmp[i, j, k] = np.dot(c_matrix, f_matrix)
