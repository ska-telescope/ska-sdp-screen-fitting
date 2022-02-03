"""
    make_aterm_images.py: Script to make a-term images from solutions

    Copyright (c) 2022, SKAO / Science Data Processor
    SPDX-License-Identifier: BSD-3-Clause
"""

import os

from ska_sdp_screen_fitting.kl_screen import KLScreen
from ska_sdp_screen_fitting.utils.h5parm import H5parm
from ska_sdp_screen_fitting.voronoi_screen import VoronoiScreen


def make_aterm_image(
    h5parmfile,
    soltabname="phase000",
    screen_type="tessellated",
    outroot="",
    bounds_deg=None,
    bounds_mid_deg=None,
    skymodel=None,
    solsetname="sol000",
    padding_fraction=1.4,
    cellsize_deg=0.2,
    smooth_deg=0,
    ncpu=0,
):
    """
    Make a-term FITS images

    Parameters
    ----------
    h5parmfile : str
        Filename of H5parm
    soltabname : str, optional
        Name of soltab to use. If "gain" is in the name, phase and amplitudes
        are used
    screen_type : str, optional
        Kind of screen to use: 'tessellated' (simple Voronoi tessellation)
        or 'kl' (Karhunen-Lo`eve transform)
    outroot : str, optional
        Root of filename of output FITS file (root+'_0.fits')
    bounds_deg : list, optional
        List of [maxRA, minDec, minRA, maxDec] for image bounds
    bounds_mid_deg : list, optional
        List of [RA, Dec] for midpoint of image bounds
    skymodel : str, optional
        Filename of calibration sky model (needed for patch positions)
    solsetname : str, optional
        Name of solset
    padding_fraction : float, optional
        Fraction of total size to pad with (e.g., 0.2 => 20% padding all
        around)
    cellsize_deg : float, optional
        Cellsize of output image
    smooth_deg : float, optional
        Size of smoothing kernel in degrees to apply
    interp_kind : str, optional
        Kind of interpolation to use. Can be any supported by
        scipy.interpolate.interp1d
    ncpu : int, optional
        Number of CPUs to use (0 means all)

    Returns
    -------
    result : dict
        Dict with list of FITS files
    """

    if "gain" in soltabname:
        # We have scalarphase and XX+YY amplitudes
        soltab_amp = soltabname.replace("gain", "amplitude")
        soltab_ph = soltabname.replace("gain", "phase")
    else:
        # We have scalarphase only
        soltab_amp = None
        soltab_ph = soltabname

    if isinstance(bounds_deg, str):
        bounds_deg = [
            float(f.strip()) for f in bounds_deg.strip("[]").split(";")
        ]
    if isinstance(bounds_mid_deg, str):
        bounds_mid_deg = [
            float(f.strip()) for f in bounds_mid_deg.strip("[]").split(";")
        ]
    if padding_fraction is not None:
        padding_fraction = float(padding_fraction)
        padding_ra = (bounds_deg[2] - bounds_deg[0]) * (padding_fraction - 1.0)
        padding_dec = (bounds_deg[3] - bounds_deg[1]) * (
            padding_fraction - 1.0
        )
        bounds_deg[0] -= padding_ra
        bounds_deg[1] -= padding_dec
        bounds_deg[2] += padding_ra
        bounds_deg[3] += padding_dec
    cellsize_deg = float(cellsize_deg)
    smooth_deg = float(smooth_deg)
    smooth_pix = smooth_deg / cellsize_deg
    if screen_type == "kl":
        # No need to smooth KL screens
        smooth_pix = 0.0

    # Check whether we just have one direction. If so, force screen_type to
    # 'tessellated' as it can handle this case and KL screens can't
    h5_file = H5parm(h5parmfile)
    solset = h5_file.get_solset(solsetname)
    soltab = solset.get_soltab(soltab_ph)
    source_names = soltab.dir[:]
    if len(source_names) == 1:
        screen_type = "tessellated"
    h5_file.close()

    # Fit screens and make a-term images
    width_deg = (
        bounds_deg[3] - bounds_deg[1]
    )  # Use Dec difference and force square images
    rootname = os.path.basename(outroot)
    if screen_type == "kl":
        screen = KLScreen(
            rootname,
            h5parmfile,
            skymodel,
            bounds_mid_deg[0],
            bounds_mid_deg[1],
            width_deg,
            width_deg,
            solset_name=solsetname,
            phase_soltab_name=soltab_ph,
            amplitude_soltab_name=soltab_amp,
        )
    elif screen_type == "tessellated":
        screen = VoronoiScreen(
            rootname,
            h5parmfile,
            skymodel,
            bounds_mid_deg[0],
            bounds_mid_deg[1],
            width_deg,
            width_deg,
            solset_name=solsetname,
            phase_soltab_name=soltab_ph,
            amplitude_soltab_name=soltab_amp,
        )
    screen.process(ncpu=ncpu)
    outdir = os.path.dirname(outroot)
    screen.write(
        outdir,
        cellsize_deg,
        smooth_pix=smooth_pix,
        ncpu=ncpu,
    )
