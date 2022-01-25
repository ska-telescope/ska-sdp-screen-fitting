import os
import shutil
import uuid
from subprocess import check_call

import pytest

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

    check_call(
        [
            "python",
            "../src/ska-sdp-screen-fitting/make_aterm_images.py",
            "--smooth_deg=0.1",
            solfile,
            f"--outroot={outroot}",
            "--bounds_deg=[126.966898;63.566717;124.546030;64.608827]",
            "--bounds_mid_deg=[125.779167;64.092778]",
            f"--skymodel={sky}",
        ]
    )

    assert os.path.isfile(f"{outroot}_0.fits")
    assert os.path.isfile(f"{outroot}_template.fits")
    assert os.path.isfile(f"{outroot}.txt")


# def test_fit_kl_screens():
#     """
#     Tests kl screens generation
#     """

#     outroot = "kl"
#     check_call(
#         [
#             "python",
#             "../src/ska-sdp-screen-fitting/make_aterm_images.py",
#             "--smooth_deg=0.1",
#             solfile,
#             f"--outroot={outroot}",
#             "--bounds_deg=[126.966898;63.566717;124.546030;64.608827]",
#             "--bounds_mid_deg=[125.779167;64.092778]",
#             "--screen_type=kl",
#             "--soltabname=gain000",
#             f"--skymodel={sky}",
#         ]
#     )

#     assert os.path.isfile(f"{outroot}_0.fits")
#     assert os.path.isfile(f"{outroot}_template.fits")
#     assert os.path.isfile(f"{outroot}.txt")
