"""
Contains class for Voronoi screens
"""

import os

import lsmtool
import numpy as np
import scipy.interpolate as si
import shapely.geometry
import shapely.ops
from astropy import wcs
from scipy.spatial import Voronoi
from shapely.geometry import Point

import ska_sdp_screen_fitting.miscellaneous as misc
from ska_sdp_screen_fitting.h5parm import H5parm
from ska_sdp_screen_fitting.screen import Screen


class VoronoiScreen(Screen):
    """
    Class for Voronoi screens
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
        super(VoronoiScreen, self).__init__(
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
        self.data_rasertize_template = None

    def fit(self):
        """
        Fitting is not needed: the input solutions are used directly, after
        referencing the phases to a single station
        """
        # Open solution tables
        H = H5parm(self.input_h5parm_filename)
        solset = H.get_solset(self.input_solset_name)
        soltab_ph = solset.get_soltab(self.input_phase_soltab_name)
        if not self.phase_only:
            soltab_amp = solset.get_soltab(self.input_amplitude_soltab_name)

        # Input data are [time, freq, ant, dir, pol] for slow amplitudes
        # and [time, freq, ant, dir] for fast phases (scalarphase).
        # We reference the phases to the station with the least amount of
        # flagged solutions, drawn from the first 10 stations
        # (to ensure it is fairly central)
        self.vals_ph = soltab_ph.val
        ref_ind = misc.get_reference_station(soltab_ph, 10)
        vals_ph_ref = self.vals_ph[:, :, ref_ind, :].copy()
        for i in range(len(soltab_ph.ant)):
            # Subtract phases of reference station
            self.vals_ph[:, :, i, :] -= vals_ph_ref
        self.times_ph = soltab_ph.time
        self.freqs_ph = soltab_ph.freq
        if not self.phase_only:
            self.log_amps = False
            self.vals_amp = soltab_amp.val
            self.times_amp = soltab_amp.time
            self.freqs_amp = soltab_amp.freq
        else:
            self.vals_amp = np.ones_like(self.vals_ph)
            self.times_amp = self.times_ph
            self.freqs_amp = self.freqs_ph

        self.source_names = soltab_ph.dir
        self.source_dict = solset.get_source()
        self.source_positions = []
        for source in self.source_names:
            self.source_positions.append(self.source_dict[source])
        self.station_names = soltab_ph.ant
        self.station_dict = solset.get_ant()
        self.station_positions = []
        for station in self.station_names:
            self.station_positions.append(self.station_dict[station])
        H.close()

    def get_memory_usage(self, cellsize_deg):
        """
        Returns memory usage per time slot in GB

        Parameters
        ----------
        cellsize_deg : float
            Size of one pixel in degrees
        """
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
            test_array.nbytes / 1024 ** 3 * 10
        )  # include factor of 10 overhead

        return mem_per_timeslot_gb

    def make_matrix(
        self,
        t_start_index,
        t_stop_index,
        freq_ind,
        stat_ind,
        cellsize_deg,
        out_dir,
        _,
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
        # Make the template that converts polynomials to a rasterized 2-D image
        # This only needs to be done once
        if self.data_rasertize_template is None:
            self.make_rasertize_template(cellsize_deg, out_dir)

        # Fill the output data array
        data = np.zeros(
            (
                t_stop_index - t_start_index,
                4,
                self.data_rasertize_template.shape[0],
                self.data_rasertize_template.shape[1],
            )
        )
        for _, poly in enumerate(self.polygons):
            ind = np.where(self.data_rasertize_template == poly.index + 1)
            if not self.phase_only:
                val_amp_xx = self.vals_amp[
                    t_start_index:t_stop_index,
                    freq_ind,
                    stat_ind,
                    poly.index,
                    0,
                ]
                val_amp_yy = self.vals_amp[
                    t_start_index:t_stop_index,
                    freq_ind,
                    stat_ind,
                    poly.index,
                    1,
                ]
            else:
                val_amp_xx = self.vals_amp[
                    t_start_index:t_stop_index, freq_ind, stat_ind, poly.index
                ]
                val_amp_yy = val_amp_xx
            val_phase = self.vals_ph[
                t_start_index:t_stop_index, freq_ind, stat_ind, poly.index
            ]
            for t in range(t_stop_index - t_start_index):
                data[t, 0, ind[0], ind[1]] = val_amp_xx[t] * np.cos(
                    val_phase[t]
                )
                data[t, 2, ind[0], ind[1]] = val_amp_yy[t] * np.cos(
                    val_phase[t]
                )
                data[t, 1, ind[0], ind[1]] = val_amp_xx[t] * np.sin(
                    val_phase[t]
                )
                data[t, 3, ind[0], ind[1]] = val_amp_yy[t] * np.sin(
                    val_phase[t]
                )

        return data

    def make_rasertize_template(self, cellsize_deg, out_dir):
        """
        Makes the template that is used to fill the output FITS cube

        Parameters
        ----------
        cellsize_deg : float
            Size of one pixel in degrees
        out_dir : str
            Full path to the output directory
        """
        temp_image = os.path.join(
            out_dir, "{}_template.fits".format(self.name)
        )
        hdu = self.make_fits_file(
            temp_image, cellsize_deg, 0, 1, aterm_type="gain"
        )
        data = hdu[0].data
        wcs_obj = wcs.WCS(hdu[0].header)
        ra_ind = wcs_obj.axis_type_names.index("RA")
        dec_ind = wcs_obj.axis_type_names.index("DEC")

        # Get x, y coords for directions in pixels. We use the input
        # calibration sky model for this, as the patch positions written to the
        # H5parm file by DPPP may  be different
        skymod = lsmtool.load(self.input_skymodel_filename)
        source_dict = skymod.getPatchPositions()
        source_positions = []
        for source in self.source_names:
            radecpos = source_dict[source.strip("[]")]
            source_positions.append([radecpos[0].value, radecpos[1].value])
        source_positions = np.array(source_positions)
        ra_deg = source_positions.T[0]
        dec_deg = source_positions.T[1]

        xy_coord = []
        for ra_vert, dec_vert in zip(ra_deg, dec_deg):
            ra_dec = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            ra_dec[0][ra_ind] = ra_vert
            ra_dec[0][dec_ind] = dec_vert
            xy_coord.append(
                (
                    wcs_obj.wcs_world2pix(ra_dec, 0)[0][ra_ind],
                    wcs_obj.wcs_world2pix(ra_dec, 0)[0][dec_ind],
                )
            )

        # Get boundary of tessellation region in pixels
        bounds_deg = [
            self.rad + self.width_ra / 2.0,
            self.dec - self.width_dec / 2.0,
            self.rad - self.width_ra / 2.0,
            self.dec + self.width_dec / 2.0,
        ]
        ra_dec = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        ra_dec[0][ra_ind] = max(bounds_deg[0], np.max(ra_deg) + 0.1)
        ra_dec[0][dec_ind] = min(bounds_deg[1], np.min(dec_deg) - 0.1)
        field_minxy = (
            wcs_obj.wcs_world2pix(ra_dec, 0)[0][ra_ind],
            wcs_obj.wcs_world2pix(ra_dec, 0)[0][dec_ind],
        )
        ra_dec[0][ra_ind] = min(bounds_deg[2], np.min(ra_deg) - 0.1)
        ra_dec[0][dec_ind] = max(bounds_deg[3], np.max(dec_deg) + 0.1)
        field_maxxy = (
            wcs_obj.wcs_world2pix(ra_dec, 0)[0][ra_ind],
            wcs_obj.wcs_world2pix(ra_dec, 0)[0][dec_ind],
        )

        if len(xy_coord) == 1:
            # If there is only a single direction, just make a single
            # rectangular polygon
            box = [
                field_minxy,
                (field_minxy[0], field_maxxy[1]),
                field_maxxy,
                (field_maxxy[0], field_minxy[1]),
                field_minxy,
            ]
            polygons = [shapely.geometry.Polygon(box)]
        else:
            # For more than one direction, tessellate
            # Generate array of outer points used to constrain the facets
            nouter = 64
            means = np.ones((nouter, 2)) * np.array(xy_coord).mean(axis=0)
            offsets = []
            angles = [np.pi / (nouter / 2.0) * i for i in range(0, nouter)]
            for ang in angles:
                offsets.append([np.cos(ang), np.sin(ang)])
            radius = 2.0 * np.sqrt(
                (field_maxxy[0] - field_minxy[0]) ** 2
                + (field_maxxy[1] - field_minxy[1]) ** 2
            )
            scale_offsets = radius * np.array(offsets)
            outer_box = means + scale_offsets

            # Tessellate and clip
            points_all = np.vstack([xy_coord, outer_box])
            vor = Voronoi(points_all)
            lines = [
                shapely.geometry.LineString(vor.vertices[line])
                for line in vor.ridge_vertices
                if -1 not in line
            ]
            polygons = [poly for poly in shapely.ops.polygonize(lines)]

        # Index polygons to directions
        for i, xypos in enumerate(xy_coord):
            for poly in polygons:
                if poly.contains(Point(xypos)):
                    poly.index = i

        # Rasterize the polygons to an array, with the value being equal to the
        # polygon's index+1
        data_template = np.ones(data[0, 0, 0, 0, :, :].shape)
        data_rasertize_template = np.zeros(data[0, 0, 0, 0, :, :].shape)
        for poly in polygons:
            verts_xy = poly.exterior.xy
            verts = []
            for x_coord, y_coord in zip(verts_xy[0], verts_xy[1]):
                verts.append((x_coord, y_coord))
            poly_raster = misc.rasterize(verts, data_template.copy()) * (
                poly.index + 1
            )
            filled = np.where(poly_raster > 0)
            data_rasertize_template[filled] = poly_raster[filled]
        zeroind = np.where(data_rasertize_template == 0)
        if len(zeroind[0]) > 0:
            nonzeroind = np.where(data_rasertize_template != 0)
            data_rasertize_template[zeroind] = si.griddata(
                (nonzeroind[0], nonzeroind[1]),
                data_rasertize_template[nonzeroind],
                (zeroind[0], zeroind[1]),
                method="nearest",
            )
        self.data_rasertize_template = data_rasertize_template
        self.polygons = polygons
