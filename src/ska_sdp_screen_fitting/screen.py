"""
    Module that holds base class for screen fitting
    SPDX-License-Identifier: BSD-3-Clause
"""
import logging
import os

import numpy as np
import scipy.interpolate as si
from astropy.coordinates import Angle
from astropy.io import fits as pyfits
from scipy import ndimage

import ska_sdp_screen_fitting.miscellaneous as misc
from ska_sdp_screen_fitting.lofar import cluster


class Screen:
    """
    Master class for a-term screens

    Parameters
    ----------
    name : str
        Name of screen
    h5parm_filename : str
        Filename of H5parm containing the input solutions
    skymodel_filename : str
        Filename of input sky model
    rad : float
        RA in degrees of screen center
    dec : float
        Dec in degrees of screen center
    width_ra : float
        Width of screen in RA in degrees, corrected to Dec = 0
    width_dec : float
        Width of screen in Dec in degrees
    solset_name: str, optional
        Name of solset of the input H5parm to use
    phase_soltab_name: str, optional
        Name of the phase soltab of the input H5parm to use
    amplitude_soltab_name: str, optional
        Name of amplitude soltab of the input H5parm to use
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
        self.name = name
        self.log = logging.getLogger(f"rapthor:{self.name}")
        self.input_h5parm_filename = h5parm_filename
        self.input_skymodel_filename = skymodel_filename
        self.input_solset_name = solset_name
        self.input_phase_soltab_name = phase_soltab_name
        self.input_amplitude_soltab_name = amplitude_soltab_name
        if self.input_amplitude_soltab_name is not None:
            self.phase_only = False
        else:
            self.phase_only = True
        if isinstance(rad, str):
            rad = Angle(rad).to("deg").value
        if isinstance(dec, str):
            dec = Angle(dec).to("deg").value
        self.rad = rad
        self.dec = dec
        width = max(
            width_ra, width_dec
        )  # force square image until rectangular ones are supported by IDG
        self.width_ra = width
        self.width_dec = width
        self.log_amps = (
            False  # sets whether amplitudes are log10 values or not
        )

        # The following attributes are assigned in the KL and Voronoi classes
        self.times_amp = None
        self.times_ph = []
        self.vals_amp = None
        self.vals_ph = None
        self.freqs_amp = None
        self.freqs_ph = None
        self.station_names = None
        self.source_names = None
        self.source_dict = None
        self.source_positions = None
        self.station_dict = None
        self.station_positions = None
        self.ncpu = None

    def fit(self):
        """
        Fits screens to the input solutions

        This method is implemented in the subclasses
        """

    def interpolate(self, interp_kind="nearest"):
        """
        Interpolate the slow amplitude values to the fast-phase time and
        frequency grid

        Parameters
        ----------
        interp_kind : str, optional
            Kind of interpolation to use
        """
        if self.phase_only:
            return

        if len(self.times_amp) == 1:
            # If only a single time, we just repeat the values as needed
            new_shape = list(self.vals_amp.shape)
            new_shape[0] = self.vals_ph.shape[0]
            new_shape[1] = self.vals_ph.shape[1]
            self.vals_amp = np.resize(self.vals_amp, new_shape)
        else:
            # Interpolate amplitudes (in log space)
            if not self.log_amps:
                logvals = np.log10(self.vals_amp)
            else:
                logvals = self.vals_amp
            if self.vals_amp.shape[0] != self.vals_ph.shape[0]:
                func = si.interp1d(
                    self.times_amp,
                    logvals,
                    axis=0,
                    kind=interp_kind,
                    fill_value="extrapolate",
                )
                logvals = func(self.times_ph)
            if self.vals_amp.shape[1] != self.vals_ph.shape[1]:
                func = si.interp1d(
                    self.freqs_amp,
                    logvals,
                    axis=1,
                    kind=interp_kind,
                    fill_value="extrapolate",
                )
                logvals = func(self.freqs_ph)
            if not self.log_amps:
                self.vals_amp = 10 ** (logvals)
            else:
                self.vals_amp = logvals

    def make_fits_file(
        self,
        outfile,
        cellsize_deg,
        t_start_index,
        t_stop_index,
        aterm_type="gain",
    ):
        """
        Makes a FITS data cube and returns the Header Data Unit

        Parameters
        ----------
        outfile : str
            Filename of output FITS file
        cellsize_deg : float
            Pixel size of image in degrees
        t_start_index : int
            Index of first time
        t_stop_index : int
            Index of last time
        aterm_type : str, optional
            Type of a-term solutions
        """
        ximsize = int(np.ceil(self.width_ra / cellsize_deg))  # pix
        yimsize = int(np.ceil(self.width_dec / cellsize_deg))  # pix
        misc.make_template_image(
            outfile,
            self.rad,
            self.dec,
            ximsize=ximsize,
            yimsize=yimsize,
            cellsize_deg=cellsize_deg,
            freqs=self.freqs_ph,
            times=self.times_ph[t_start_index:t_stop_index],
            antennas=self.station_names,
            aterm_type=aterm_type,
        )
        hdu = pyfits.open(outfile, memmap=False)
        return hdu

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

        This method should be defined in the subclasses, but should conform to
        the inputs below.

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
            Full path to the output directory (needed for template file
            generation)
        ncpu : int, optional
            Number of CPUs to use (0 means all)
        """

        # Dummy implementation to avoid pylint errors
        del (
            t_start_index,
            t_stop_index,
            freq_ind,
            stat_ind,
            cellsize_deg,
            out_dir,
            ncpu,
        )
        data = None
        print(f"This function should never be entered in {self.name}")
        return data

    def get_memory_usage(self, cellsize_deg):
        """
        Returns memory usage per time slot in GB

        This method should be defined in the subclasses, but should conform to
        the inputs below.

        Parameters
        ----------
        cellsize_deg : float
            Size of one pixel in degrees
        """

    def write(
        self,
        out_dir,
        cellsize_deg,
        smooth_pix=0,
        ncpu=0,
    ):
        """
        Write the a-term screens to a FITS data cube

        Parameters
        ----------
        out_dir : str
            Output directory
        cellsize_deg : float
            Size of one pixel in degrees
        smooth_pix : int, optional
            Size of Gaussian in pixels to smooth with
        ncpu : int, optional
            Number of CPUs to use (0 means all)
        """
        self.ncpu = ncpu

        # Identify any gaps in time (frequency gaps are not allowed), as we
        # need to output a separate FITS file for each time chunk
        if len(self.times_ph) > 2:
            delta_times = (
                self.times_ph[1:] - self.times_ph[:-1]
            )  # time at center of solution interval
            timewidth = np.min(delta_times)
            gaps = np.where(delta_times > timewidth * 1.2)
            gaps_ind = gaps[0] + 1
            gaps_ind = np.append(gaps_ind, np.array([len(self.times_ph)]))
        else:
            gaps_ind = np.array([len(self.times_ph)])

        # Add additional breaks to gaps_ind to keep memory usage within that
        #  available
        if len(self.times_ph) > 2:
            available_mem_gb = cluster.get_available_memory()
            max_ntimes = max(
                1,
                int(available_mem_gb / (self.get_memory_usage(cellsize_deg))),
            )
            check_gaps = True
            while check_gaps:
                check_gaps = False
                g_start = 0
                gaps_ind_copy = gaps_ind.copy()
                for gnum, g_stop in enumerate(gaps_ind_copy):
                    if g_stop - g_start > max_ntimes:
                        new_gap = g_start + int((g_stop - g_start) / 2)
                        gaps_ind = np.insert(
                            gaps_ind, gnum, np.array([new_gap])
                        )
                        check_gaps = True
                        break
                    g_start = g_stop

        # Input data are [time, freq, ant, dir, pol] for slow amplitudes
        # and [time, freq, ant, dir] for fast phases (scalarphase).
        # Output data are [RA, DEC, MATRIX, ANTENNA, FREQ, TIME].T
        # Loop over stations, frequencies, and times and fill in the correct
        # matrix values (matrix dimension has 4 elements: real XX,
        # imaginary XX, real YY and imaginary YY)
        outroot = self.name
        outfiles = []
        g_start = 0
        for gnum, g_stop in enumerate(gaps_ind):
            ntimes = g_stop - g_start
            outfile = os.path.join(out_dir, f"{outroot}_{gnum}.fits")
            hdu = self.make_fits_file(
                outfile, cellsize_deg, g_start, g_stop, aterm_type="gain"
            )
            data = hdu[0].data
            for freq_, _ in enumerate(self.freqs_ph):
                for station, _ in enumerate(self.station_names):
                    print(
                        "Writing freq: "
                        + str(freq_)
                        + ", station: "
                        + str(station)
                    )
                    data[:, freq_, station, :, :, :] = self.make_matrix(
                        g_start,
                        g_stop,
                        freq_,
                        station,
                        cellsize_deg,
                        out_dir,
                        self.ncpu,
                    )

                    # Smooth if desired
                    if smooth_pix > 0:
                        for time in range(ntimes):
                            data[
                                time, freq_, station, :, :, :
                            ] = ndimage.gaussian_filter(
                                data[time, freq_, station, :, :, :],
                                sigma=(0, smooth_pix, smooth_pix),
                                order=0,
                            )

            # Ensure there are no NaNs in the images, as WSClean will produced
            # uncorrected, uncleaned images if so. We replace NaNs with 1.0 and
            # 0.0 for real and imaginary parts, respectively
            # Note: we iterate over time to reduce memory usage
            for time in range(ntimes):
                for p_val in range(4):
                    if p_val % 2:
                        # Imaginary elements
                        nanval = 0.0
                    else:
                        # Real elements
                        nanval = 1.0
                    data[time, :, :, p_val, :, :][
                        np.isnan(data[time, :, :, p_val, :, :])
                    ] = nanval

            # Write FITS file
            hdu[0].data = data
            hdu.writeto(outfile, overwrite=True)
            outfiles.append(outfile)
            hdu = None
            data = None

            # Update start time index before starting next loop
            g_start = g_stop

        # Write list of filenames to a text file for later use
        with open(
            os.path.join(out_dir, f"{outroot}.txt"), "w", encoding="utf8"
        ) as list_file:
            list_file.writelines([o + "\n" for o in outfiles])

    def process(self, ncpu=0):
        """
        Makes a-term images

        Parameters
        ----------
        ncpu : int, optional
            Number of CPUs to use (0 means all)
        """
        self.ncpu = ncpu

        # Fit screens to input solutions
        self.fit()

        # Interpolate best-fit parameters to common time and frequency grid
        self.interpolate()
