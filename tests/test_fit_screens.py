""" Test screen fitting functionality """

import os
import shutil
import sys
import uuid

import pytest

sys.path.append("./src/ska_sdp_screen_fitting")
from make_aterm_images import main  # NOQA: E402, C0413

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

    outroot = "tessellated"
    main(
        SOLFILE,
        "phase000",
        "tessellated",
        outroot,
        [126.966898, 63.566717, 124.546030, 64.608827],
        [125.779167, 64.092778],
        SKYMODEL,
        "sol000",
        1.4,
        0.2,
        0.1,
        0,
    )

    assert os.path.isfile(f"{outroot}_0.fits")
    assert os.path.isfile(f"{outroot}_template.fits")
    assert os.path.isfile(f"{outroot}.txt")


def test_fit_kl_screens():
    """
    Tests kl screens generation
    """

    outroot = "kl"
    main(
        SOLFILE,
        "gain000",
        "kl",
        outroot,
        [126.966898, 63.566717, 124.546030, 64.608827],
        [125.779167, 64.092778],
        SKYMODEL,
        "sol000",
        1.4,
        0.2,
        0.1,
        0,
    )

    assert os.path.isfile(f"{outroot}_0.fits")
    assert os.path.isfile(f"{outroot}.txt")
