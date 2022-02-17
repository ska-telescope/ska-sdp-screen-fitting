"""
test_fit_screens.py: Test screen fitting functionality
SPDX-License-Identifier: BSD-3-Clause
"""

import os
import shutil
import uuid

import h5py
import lsmtool
import numpy as np
import pytest
from astropy import wcs
from astropy.io import fits

from ska_sdp_screen_fitting.make_aterm_images import make_aterm_image

CWD = os.getcwd()
SOLFILE = "solutions.h5"
SKYMODEL = "skymodel.txt"


@pytest.fixture(autouse=True)
def source_env():
    """Create temporary folder for test"""
    os.chdir(CWD)
    tmpdir = str(uuid.uuid4())
    os.mkdir(tmpdir)
    os.chdir(tmpdir)

    shutil.copyfile(f"../resources/{SOLFILE}", SOLFILE)
    shutil.copyfile(f"../resources/{SKYMODEL}", SKYMODEL)

    # Tests are executed here
    yield

    # Post-test: clean up
    os.chdir(CWD)
    shutil.rmtree(tmpdir)


def read_patch_list(skymodel, h5_file, soltab):
    """Read patch coordinates from skymodel file"""

    skymod = lsmtool.load(skymodel)
    source_dict = skymod.getPatchPositions()
    source_positions = []
    for source in list(h5_file[f"sol000/{soltab}/dir"]):
        radecpos = source_dict[str(source, "utf-8").strip("[]")]
        source_positions.append([radecpos[0].value, radecpos[1].value])
    source_positions = np.array(source_positions)
    return source_positions


def get_patch_coordinates(source_positions, wcs_obj):
    """Convert coordinates from ra-dec to pixel coordinates"""

    ra_ind = wcs_obj.axis_type_names.index("RA")
    dec_ind = wcs_obj.axis_type_names.index("DEC")

    # Convert ra-dec to pixel coordinates
    ra = source_positions[:, 0]
    dec = source_positions[:, 1]
    xy_coord = []
    coord_patch_x = []
    coord_patch_y = []
    for ra_vert, dec_vert in zip(ra, dec):
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


def test_fit_voronoi_screens():
    """
    Tests Voronoi screens generation
    """

    method = "tessellated"
    soltab = "phase000"
    make_aterm_image(
        SOLFILE,
        soltabname=soltab,
        screen_type=method,
        outroot=method,
        bounds_deg=[126.966898, 63.566717, 124.546030, 64.608827],
        bounds_mid_deg=[125.779167, 64.092778],
        skymodel=SKYMODEL,
        solsetname="sol000",
        padding_fraction=1.4,
        cellsize_deg=0.2,
        smooth_deg=0,
        ncpu=0,
    )

    # Assert that solution files are generated
    assert os.path.isfile(f"{method}_0.fits")
    assert os.path.isfile(f"{method}_template.fits")
    assert os.path.isfile(f"{method}.txt")

    # Load h5 solutions and image cube and calculate the error at the
    # patch coordinates
    # 1 - Get the pixel coordinate of the patches
    # 2 - Open the calibration solution and correct for the phase reference
    h5_file = h5py.File(SOLFILE, "r")
    radec_coord = read_patch_list(SKYMODEL, h5_file, soltab)
    filename = f"{method}_0.fits"
    hdu = fits.open(filename)
    wcs_obj = wcs.WCS(hdu[0].header)
    [coord_x, coord_y] = get_patch_coordinates(radec_coord, wcs_obj)
    screen_cube = hdu[0].data
    im_size = screen_cube.shape[4]
    phase = h5_file["sol000/phase000/val"]
    phase_corrected = np.zeros(
        (
            screen_cube.shape[0],
            screen_cube.shape[1],
            screen_cube.shape[2],
            len(radec_coord),
        )
    )
    for i in range(phase_corrected.shape[0]):
        for k in range(phase_corrected.shape[1]):
            for j in range(phase_corrected.shape[3]):
                phase_corrected[i, k, :, j] = (
                    phase[i, k, :, j] - phase[i, k, 0, j]
                )

    # Assert that the error at the position of the patch is smaller
    # than the threshold
    threshold = 1e-4
    for i in enumerate(coord_x):
        if coord_x[i[0]] >= 0 and coord_x[i[0]] < im_size:
            if coord_y[i[0]] >= 0 and coord_y[i[0]] < im_size:
                assert (
                    screen_cube[:, :, :, 0, coord_x[i[0]], coord_y[i[0]]]
                    - np.cos(phase_corrected[:, :, :, i[0]])
                    < threshold
                ).all()
                assert (
                    screen_cube[:, :, :, 1, coord_x[i[0]], coord_y[i[0]]]
                    - np.sin(phase_corrected[:, :, :, i[0]])
                    < threshold
                ).all()
                assert (
                    screen_cube[:, :, :, 2, coord_x[i[0]], coord_y[i[0]]]
                    - np.cos(phase_corrected[:, :, :, i[0]])
                    < threshold
                ).all()
                assert (
                    screen_cube[:, :, :, 3, coord_x[i[0]], coord_y[i[0]]]
                    - np.sin(phase_corrected[:, :, :, i[0]])
                    < threshold
                ).all()


def test_fit_kl_screens():
    """
    Tests kl screens generation
    """

    method = "kl"
    make_aterm_image(
        SOLFILE,
        soltabname="gain000",
        screen_type=method,
        outroot=method,
        bounds_deg=[126.966898, 63.566717, 124.546030, 64.608827],
        bounds_mid_deg=[125.779167, 64.092778],
        skymodel=SKYMODEL,
        solsetname="sol000",
        padding_fraction=1.4,
        cellsize_deg=0.2,
        smooth_deg=0,
        ncpu=0,
    )

    # Assert that solution files are generated
    assert os.path.isfile(f"{method}_0.fits")
    assert os.path.isfile(f"{method}.txt")

    # Load h5 solutions and image cube and calculate the error at the
    # patch coordinates
    # 1 - Get the pixel coordinate of the patches
    # 2 - Open the calibration solution and correct for the phase reference
    h5_file = h5py.File(SOLFILE, "r")
    radec_coord = read_patch_list(SKYMODEL, h5_file, "phase000")
    filename = f"{method}_0.fits"
    hdu = fits.open(filename)
    wcs_obj = wcs.WCS(hdu[0].header)
    [coord_x, coord_y] = get_patch_coordinates(radec_coord, wcs_obj)

    screen_cube = hdu[0].data
    im_size = screen_cube.shape[4]
    ampl = h5_file["sol000/amplitude000/val"]  # [1, 50, 62, 7, 2]
    phase = h5_file["sol000/phase000/val"]  # [3,50,62,7]

    phase_corrected = np.zeros(
        (
            screen_cube.shape[0],
            screen_cube.shape[1],
            screen_cube.shape[2],
            len(radec_coord),
        )
    )
    for i in range(phase_corrected.shape[0]):
        for k in range(phase_corrected.shape[1]):
            for j in range(phase_corrected.shape[3]):
                phase_corrected[i, k, :, j] = (
                    phase[i, k, :, j] - phase[i, k, 0, j]
                )

    # Assert that the error at the position of the patch is smaller
    # than the threshold
    threshold = 1e-2
    for i in enumerate(coord_x):
        if coord_x[i[0]] >= 0 and coord_x[i[0]] < im_size:
            if coord_y[i[0]] >= 0 and coord_y[i[0]] < im_size:
                assert (
                    screen_cube[:, :, :, 0, coord_x[i[0]], coord_y[i[0]]]
                    - ampl[0, :, :, i[0], 0]
                    * np.cos(phase_corrected[:, :, :, i[0]])
                    < threshold
                ).all()
                assert (
                    screen_cube[:, :, :, 1, coord_x[i[0]], coord_y[i[0]]]
                    - ampl[0, :, :, i[0], 0]
                    * np.sin(phase_corrected[:, :, :, i[0]])
                    < threshold
                ).all()
                assert (
                    screen_cube[:, :, :, 2, coord_x[i[0]], coord_y[i[0]]]
                    - ampl[0, :, :, i[0], 1]
                    * np.cos(phase_corrected[:, :, :, i[0]])
                    < threshold
                ).all()
                assert (
                    screen_cube[:, :, :, 3, coord_x[i[0]], coord_y[i[0]]]
                    - ampl[0, :, :, i[0], 1]
                    * np.sin(phase_corrected[:, :, :, i[0]])
                    < threshold
                ).all()
