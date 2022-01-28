""" Test screen fitting functionality """

import os
import shutil
import uuid
from subprocess import check_call

import pytest

CWD = os.getcwd()
SOLFILE = "solutions.h5"
SKYMODEL = "skymodel.txt"
SCREEN_FITTING = "ska-sdp-screen-fitting"

COMMON_ARGS = [
    "--bounds_deg=[126.966898;63.566717;124.546030;64.608827]",
    "--bounds_mid_deg=[125.779167;64.092778]",
    f"--skymodel={SKYMODEL}",
    "--solsetname=sol000",
    "--padding_fraction=1.4",
    "--cellsize_deg=0.2",
    "--smooth_deg=0.1",
    "--ncpu=0",
]


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

    outroot = "tessellated"
    check_call(
        [
            SCREEN_FITTING,
            SOLFILE,
            "--soltabname=phase000",
            f"--screen_type={outroot}",
            f"--outroot={outroot}",
        ]
        + COMMON_ARGS
    )

    assert os.path.isfile(f"{outroot}_0.fits")
    assert os.path.isfile(f"{outroot}_template.fits")
    assert os.path.isfile(f"{outroot}.txt")


def test_fit_kl_screens():
    """
    Tests kl screens generation
    """

    outroot = "kl"
    check_call(
        [
            SCREEN_FITTING,
            SOLFILE,
            "--soltabname=gain000",
            f"--screen_type={outroot}",
            f"--outroot={outroot}",
        ]
        + COMMON_ARGS
    )

    assert os.path.isfile(f"{outroot}_0.fits")
    assert os.path.isfile(f"{outroot}.txt")
