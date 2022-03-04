"""
test_fit_screens.py: Test screen fitting functionality
SPDX-License-Identifier: BSD-3-Clause
"""

import os
import shutil
import uuid

import h5py
import numpy as np
import pytest
from astropy import wcs
from astropy.io import fits

from ska_sdp_screen_fitting.make_aterm_images import make_aterm_image
from ska_sdp_screen_fitting.utils import processing_utils

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
        bounds_deg=[124.565, 66.165, 127.895, 62.835],
        bounds_mid_deg=[126.23, 64.50],
        skymodel=SKYMODEL,
        solsetname="sol000",
        padding_fraction=0,
        cellsize_deg=0.2,
        smooth_deg=0.1,
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
    radec_coord = processing_utils.read_patch_list(SKYMODEL, h5_file, soltab)
    filename = f"{method}_0.fits"
    hdu = fits.open(filename)
    wcs_obj = wcs.WCS(hdu[0].header)
    [coord_x, coord_y] = processing_utils.get_patch_coordinates(
        radec_coord, wcs_obj
    )
    screen_cube = hdu[0].data
    im_size = screen_cube.shape[4]
    phase = h5_file["sol000/phase000/val"]

    # re-arrange axes to allow correct broadcasting
    ref_antenna = 0
    phase_corrected = np.zeros(
        (
            screen_cube.shape[0],
            screen_cube.shape[1],
            screen_cube.shape[2],
            len(radec_coord),
        )
    )
    phase_corrected = (
        np.transpose(phase, (2, 0, 1, 3)) - phase[:, :, ref_antenna, :]
    )
    phase_corrected = np.transpose(phase_corrected, (1, 2, 0, 3))

    # Assert that the error at the position of the patch is smaller
    # than the threshold
    threshold = 1e-4
    for i in enumerate(coord_x):
        y = int(np.round(coord_x[i[0]]))
        x = int(np.round(coord_y[i[0]]))
        if x >= 0 and x < im_size:
            if y >= 0 and y < im_size:
                assert (
                    screen_cube[:, :, :, 0, x, y]
                    - np.cos(phase_corrected[:, :, :, i[0]])
                    < threshold
                ).all()
                assert (
                    screen_cube[:, :, :, 1, x, y]
                    - np.sin(phase_corrected[:, :, :, i[0]])
                    < threshold
                ).all()
                assert (
                    screen_cube[:, :, :, 2, x, y]
                    - np.cos(phase_corrected[:, :, :, i[0]])
                    < threshold
                ).all()
                assert (
                    screen_cube[:, :, :, 3, x, y]
                    - np.sin(phase_corrected[:, :, :, i[0]])
                    < threshold
                ).all()


def test_fit_kl_screens():
    """
    Tests kl screens generation
    """

    soltab = "phase000"
    method = "kl"
    make_aterm_image(
        SOLFILE,
        soltabname=soltab,
        screen_type=method,
        outroot=method,
        bounds_deg=[124.565, 66.165, 127.895, 62.835],
        bounds_mid_deg=[126.23, 64.50],
        skymodel=SKYMODEL,
        solsetname="sol000",
        padding_fraction=0,
        cellsize_deg=0.2,
        smooth_deg=0.1,
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
    radec_coord = processing_utils.read_patch_list(SKYMODEL, h5_file, soltab)
    filename = f"{method}_0.fits"
    hdu = fits.open(filename)
    wcs_obj = wcs.WCS(hdu[0].header)
    [coord_x, coord_y] = processing_utils.get_patch_coordinates(
        radec_coord, wcs_obj
    )

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
    ref_antenna = 0
    phase_corrected = (
        np.transpose(phase, (2, 0, 1, 3)) - phase[:, :, ref_antenna, :]
    )
    phase_corrected = np.transpose(phase_corrected, (1, 2, 0, 3))

    # Assert that the error at the position of the patch is smaller
    # than the threshold
    threshold = 1e-1
    for i in enumerate(coord_x):
        y = int(np.round(coord_x[i[0]]))
        x = int(np.round(coord_y[i[0]]))
        if x >= 0 and x < im_size:
            if y >= 0 and y < im_size:
                assert (
                    screen_cube[:, :, :, 0, x, y]
                    - np.cos(phase_corrected[:, :, :, i[0]])
                    < threshold
                ).all()
                assert (
                    screen_cube[:, :, :, 1, x, y]
                    - np.sin(phase_corrected[:, :, :, i[0]])
                    < threshold
                ).all()
                assert (
                    screen_cube[:, :, :, 2, x, y]
                    - np.cos(phase_corrected[:, :, :, i[0]])
                    < threshold
                ).all()
                assert (
                    screen_cube[:, :, :, 3, x, y]
                    - np.sin(phase_corrected[:, :, :, i[0]])
                    < threshold
                ).all()
