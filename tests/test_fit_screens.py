import os
import shutil
import sys
import uuid

import pytest

sys.path.append("./src/ska_sdp_screen_fitting")
from make_aterm_images import main

""" Test screen functionality """

CWD = os.getcwd()
solfile = "solutions.h5"
sky = "skymodel.txt"


@pytest.fixture(autouse=True)
def source_env():
    os.chdir(CWD)
    tmpdir = str(uuid.uuid4())
    os.mkdir(tmpdir)
    os.chdir(tmpdir)

    shutil.copyfile(f"../resources/{solfile}", solfile)
    shutil.copyfile(f"../resources/{sky}", sky)

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
        solfile,
        "phase000",
        "tessellated",
        outroot,
        [126.966898, 63.566717, 124.546030, 64.608827],
        [125.779167, 64.092778],
        sky,
        "sol000",
        1.4,
        0.2,
        0.1,
        "nearest",
        0,
    )

    assert os.path.isfile(f"{outroot}_0.fits")
    assert os.path.isfile(f"{outroot}_template.fits")
    assert os.path.isfile(f"{outroot}.txt")


# # test disabled because longer than timeout
# def test_fit_kl_screens():
#     """
#     Tests kl screens generation
#     """

#     outroot = "kl"
#     main(
#         solfile,
#         "gain000",
#         "kl",
#         outroot,
#         [126.966898, 63.566717, 124.546030, 64.608827],
#         [125.779167, 64.092778],
#         sky,
#         "sol000",
#         1.4,
#         0.2,
#         0.1,
#         "nearest",
#         0,
#     )

#     assert os.path.isfile(f"{outroot}_0.fits")
#     assert os.path.isfile(f"{outroot}_template.fits")
#     assert os.path.isfile(f"{outroot}.txt")
