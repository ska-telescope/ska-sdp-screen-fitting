import itertools
import multiprocessing
from multiprocessing import Pool, RawArray

import lsmtool
import miscellaneous as misc
import numpy as np
import stationscreen
from astropy import wcs
from h5parm import h5parm
from screen import Screen


class KLScreen(Screen):
    """
    Class for KL (Karhunen-Lo`eve) screens
    """

    def __init__(
        self,
        name,
        h5parm_filename,
        skymodel_filename,
        ra,
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
            ra,
            dec,
            width_ra,
            width_dec,
            solset_name=solset_name,
            phase_soltab_name=phase_soltab_name,
            amplitude_soltab_name=amplitude_soltab_name,
        )

    def fit(self):
        """
        Fits screens to the input solutions
        """
        # Open solution tables
        H = h5parm(self.input_h5parm_filename, readonly=False)
        solset = H.getSolset(self.input_solset_name)
        soltab_ph = solset.getSoltab(self.input_phase_soltab_name)
        if not self.phase_only:
            soltab_amp = solset.getSoltab(self.input_amplitude_soltab_name)

        # Set the position of the calibration patches to those of
        # the input sky model, as the patch positions written to the h5parm
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
        #     [ra * np.pi / 180.0, dec * np.pi / 180.0]
        #     for ra, dec in zip(ra_deg, dec_deg)
        # ]
        # sourceTable = list(zip(*(soltab_ph.dir, vals)))

        # Now call LoSoTo's stationscreen operation to do the fitting. For the
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
            refAnt=ref_ind,
            scale_order=True,
            adjust_order=adjust_order_ph,
            ncpu=self.ncpu,
        )
        soltab_ph_screen = solset.getSoltab("phase_screen000")
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
            soltab_amp_screen = solset.getSoltab("amplitude_screen000")
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
        self.source_dict = solset.getSou()
        self.source_positions = []
        for source in self.source_names:
            self.source_positions.append(self.source_dict[source])
        self.station_names = soltab_ph_screen.ant
        self.station_dict = solset.getAnt()
        self.station_positions = []
        for station in self.station_names:
            self.station_positions.append(self.station_dict[station])
        self.height = soltab_ph_screen.obj._v_attrs["height"]
        self.beta_val = soltab_ph_screen.obj._v_attrs["beta"]
        self.r_0 = soltab_ph_screen.obj._v_attrs["r_0"]
        self.pp = np.array(soltab_ph_screen.obj.piercepoint)
        self.midRA = soltab_ph_screen.obj._v_attrs["midra"]
        self.midDec = soltab_ph_screen.obj._v_attrs["middec"]
        H.close()

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
            test_array.nbytes / 1024 ** 3 / 10
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
        out_dir,
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
        global screen_ph, screen_amp_xx, screen_amp_yy, pp, x, y, var_dict

        # Define various parameters
        N_sources = len(self.source_names)
        N_times = t_stop_index - t_start_index
        N_piercepoints = N_sources
        beta_val = self.beta_val
        r_0 = self.r_0

        # Make arrays of pixel coordinates for screen
        # We need to convert the FITS cube pixel coords to screen pixel coords.
        # The FITS cube has self.ra, self.dec at (xsize/2, ysize/2)
        ximsize = int(np.ceil(self.width_ra / cellsize_deg))  # pix
        yimsize = int(np.ceil(self.width_dec / cellsize_deg))  # pix
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [ximsize / 2.0, yimsize / 2.0]
        w.wcs.cdelt = np.array([-cellsize_deg, cellsize_deg])
        w.wcs.crval = [self.ra, self.dec]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.set_pv([(2, 1, 45.0)])

        x_fits = list(range(ximsize))
        y_fits = list(range(yimsize))
        ra = []
        dec = []
        for xf, yf in zip(x_fits, y_fits):
            x_y = np.array([[xf, yf]])
            ra.append(w.wcs_pix2world(x_y, 0)[0][0])
            dec.append(w.wcs_pix2world(x_y, 0)[0][1])
        xy, _, _ = stationscreen._getxy(
            ra, dec, midRA=self.midRA, midDec=self.midDec
        )
        x = xy[0].T
        y = xy[1].T
        Nx = len(x)
        Ny = len(y)

        # Select input data and reorder the axes to get axis order of
        # [dir, time]
        # Input data are [time, freq, ant, dir, pol] for slow amplitudes
        # and [time, freq, ant, dir] for fast phases (scalarphase).
        time_axis = 0
        dir_axis = 1
        screen_ph = np.array(
            self.vals_ph[t_start_index:t_stop_index, freq_ind, stat_ind, :]
        )
        screen_ph = screen_ph.transpose([dir_axis, time_axis])
        if not self.phase_only:
            screen_amp_xx = np.array(
                self.vals_amp[
                    t_start_index:t_stop_index, freq_ind, stat_ind, :, 0
                ]
            )
            screen_amp_xx = screen_amp_xx.transpose([dir_axis, time_axis])
            screen_amp_yy = np.array(
                self.vals_amp[
                    t_start_index:t_stop_index, freq_ind, stat_ind, :, 1
                ]
            )
            screen_amp_yy = screen_amp_yy.transpose([dir_axis, time_axis])

        # Process phase screens
        ncpu = self.ncpu
        if ncpu == 0:
            ncpu = multiprocessing.cpu_count()
        pp = self.pp
        val_shape = (Nx, Ny, N_times)
        var_dict = {}
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
                    itertools.repeat(N_piercepoints),
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
                        itertools.repeat(N_piercepoints),
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
                        itertools.repeat(N_piercepoints),
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
        data = np.zeros((N_times, 4, Ny, Nx))
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
    for storing store shared_val and val_shape in var_dict (a global variable).

    See https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html # NOQA: E501

    Parameters
    ----------
    shared_val : array
        RawArray to be shared
    val_shape : tuple
        Shape of shared_val array
    """
    global var_dict

    var_dict["shared_val"] = shared_val
    var_dict["val_shape"] = val_shape


def calculate_kl_screen_star(inputs):
    """
    Simple helper function for pool.map
    """
    return calculate_kl_screen(*inputs)


def calculate_kl_screen(k, N_piercepoints, beta_val, r_0, screen_type):
    """
    Calculates screen images

    Parameters
    ----------
    k : int
        Time index
    N_piercepoints : int
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
    global screen_ph, screen_amp_xx, screen_amp_yy, pp, x, y, var_dict

    tmp = np.frombuffer(var_dict["shared_val"], dtype=np.float64).reshape(
        var_dict["val_shape"]
    )
    if screen_type == "ph":
        inscreen = screen_ph[:, k]
    if screen_type == "xx":
        inscreen = screen_amp_xx[:, k]
    if screen_type == "yy":
        inscreen = screen_amp_yy[:, k]
    f = inscreen.reshape(N_piercepoints)
    for i, xi in enumerate(x):
        for j, yi in enumerate(y):
            p = np.array([xi, yi, 0.0])
            d2 = np.sum(np.square(pp - p), axis=1)
            c = -((d2 / (r_0 ** 2)) ** (beta_val / 2.0)) / 2.0
            tmp[i, j, k] = np.dot(c, f)
