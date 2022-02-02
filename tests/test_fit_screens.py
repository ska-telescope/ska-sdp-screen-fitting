""" Test screen fitting functionality """

import os
import shutil
import uuid

import pytest

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


def test_fit_voronoi_screens():
    """
    Tests Voronoi screens generation
    """

    method = "tessellated"
    make_aterm_image(
        SOLFILE,
        soltabname="phase000",
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
