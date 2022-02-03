"""
    main.py: Script to invoke screen fitting algorithm

    Copyright (c) 2022, SKAO / Science Data Processor
    SPDX-License-Identifier: BSD-3-Clause
"""

import argparse

from ska_sdp_screen_fitting.make_aterm_images import make_aterm_image


def start():
    """
    This is the entry point for the executable
    """

    description_text = "Make a-term images from solutions.\n"

    parser = argparse.ArgumentParser(
        description=description_text,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("h5parmfile", help="Filename of input H5parm")
    parser.add_argument(
        "--soltabname", help="Name of soltab", type=str, default="phase000"
    )
    parser.add_argument(
        "--screen_type",
        help="Type of screen",
        type=str,
        default="tessellated",
    )
    parser.add_argument(
        "--outroot", help="Root of output images", type=str, default=""
    )
    parser.add_argument(
        "--bounds_deg", help="Bounds list in deg", type=str, default=None
    )
    parser.add_argument(
        "--bounds_mid_deg",
        help="Bounds mid list in deg",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--skymodel", help="Filename of sky model", type=str, default=None
    )
    parser.add_argument(
        "--solsetname", help="Solset name", type=str, default="sol000"
    )
    parser.add_argument(
        "--padding_fraction", help="Padding fraction", type=float, default=1.4
    )
    parser.add_argument(
        "--cellsize_deg", help="Cell size in deg", type=float, default=0.2
    )
    parser.add_argument(
        "--smooth_deg", help="Smooth scale in degree", type=float, default=0.0
    )
    parser.add_argument(
        "--ncpu", help="Number of CPUs to use", type=int, default=0
    )
    args = parser.parse_args()
    make_aterm_image(
        args.h5parmfile,
        soltabname=args.soltabname,
        screen_type=args.screen_type,
        outroot=args.outroot,
        bounds_deg=args.bounds_deg,
        bounds_mid_deg=args.bounds_mid_deg,
        skymodel=args.skymodel,
        solsetname=args.solsetname,
        padding_fraction=args.padding_fraction,
        cellsize_deg=args.cellsize_deg,
        smooth_deg=args.smooth_deg,
        ncpu=args.ncpu,
    )
