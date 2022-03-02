"""Script to plot screens with input solutions overlay"""

import h5py
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy import wcs
import numpy as np

import sys

sys.path.insert(1, "../src/ska_sdp_screen_fitting/utils")
import processing_utils


def build_input_grid(
    ntime, nfreq, nantennas, npol, im_size, patch_idx, coord_patch_x, coord_patch_y
):

    """Creates grid with input values
    Parameters
    ----------
    ntime : int
        Number of solution intervals
    nfreq : int
        Number of frequency channels
    nantennas : int
        Number of antennas
    npol : int
        Number of polarizations
    im_size : int
        Screen width in pixels (width = height)
    patch_idx : numpy array
        Array containing the indexes of the skymodel points to use
    coord_patch_x : numpy array
        X coordinates of skymodel points
    coord_patch_y : numpy array
        Y coordinates of skymodel points

    Returns
    -------
    grid_kl : ndarray
        Cube containing the kl solution value at the skymodel points, zeroes elsewhere
    grid_voronoi : ndarray
        Cube containing the voronoi solution value at the skymodel points, zeroes elsewhere
    patch_mask : ndarray
        Cube containing ones at the skymodel points, zeroes elsewhere
    """

    grid_kl = np.zeros((ntime, nfreq, nantennas, npol, im_size, im_size))
    grid_voronoi = np.zeros((ntime, nfreq, nantennas, npol, im_size, im_size))

    patch_mask = np.full((im_size, im_size), False)

    for t in range(ntime):
        for i in range(patch_idx.size):
            x_idx = coord_patch_x[patch_idx[i]]
            y_idx = coord_patch_y[patch_idx[i]]

            grid_kl[t, :, :, 0, x_idx, y_idx] = ampl[0, :, :, patch_idx[i], 0] * np.cos(
                phase_corrected[t, :, :, patch_idx[i]]
            )
            grid_kl[t, :, :, 2, x_idx, y_idx] = ampl[0, :, :, patch_idx[i], 1] * np.cos(
                phase_corrected[t, :, :, patch_idx[i]]
            )
            grid_kl[t, :, :, 1, x_idx, y_idx] = ampl[0, :, :, patch_idx[i], 0] * np.sin(
                phase_corrected[t, :, :, patch_idx[i]]
            )
            grid_kl[t, :, :, 3, x_idx, y_idx] = ampl[0, :, :, patch_idx[i], 1] * np.sin(
                phase_corrected[t, :, :, patch_idx[i]]
            )

            grid_voronoi[t, :, :, 0, x_idx, y_idx] = np.cos(
                phase_corrected[t, :, :, patch_idx[i]]
            )
            grid_voronoi[t, :, :, 2, x_idx, y_idx] = np.cos(
                phase_corrected[t, :, :, patch_idx[i]]
            )
            grid_voronoi[t, :, :, 1, x_idx, y_idx] = np.sin(
                phase_corrected[t, :, :, patch_idx[i]]
            )
            grid_voronoi[t, :, :, 3, x_idx, y_idx] = np.sin(
                phase_corrected[t, :, :, patch_idx[i]]
            )

            patch_mask[x_idx, y_idx] = True

    return grid_kl, grid_voronoi, patch_mask


def get_boundaries(
    kl_cube,
    voronoi_cube,
    grid,
    time,
    freq_offset,
    n_subplots,
    antenna,
    polarization_idx,
):

    """Calculates min/max values of the input cubes in the selected range
    Parameters
    ----------
    kl_cube : ndarray
        Cube containing the kl screen solutions
    voronoi_cube : ndarray
        Cube containing the voronoi screen solutions
    grid : ndarray
        Cube containing the solution value at the skymodel points, zeroes elsewhere
    time : int
        Number of solution intervals
    freq_offset : int
        Number of frequency channels
    n_subplots : int
        Number of antennas
    antenna : int
        Number of polarizations
    polarization_idx : int
        Screen width in pixels (width = height)

    Returns
    -------
    min_val : ndarray
        Min value between the input cubes within the selected range
    max_val : ndarray
        Max value between the input cubes within the selected range
    """

    kl_min = np.min(
        kl_cube.data[
            time,
            freq_offset : freq_offset + n_subplots,
            antenna,
            polarization_idx,
            :,
            :,
        ]
    )
    kl_max = np.max(
        kl_cube.data[
            time,
            freq_offset : freq_offset + n_subplots,
            antenna,
            polarization_idx,
            :,
            :,
        ]
    )
    vor_min = np.min(
        voronoi_cube.data[
            time,
            freq_offset : freq_offset + n_subplots,
            antenna,
            polarization_idx,
            :,
            :,
        ]
    )
    vor_max = np.max(
        voronoi_cube.data[
            time,
            freq_offset : freq_offset + n_subplots,
            antenna,
            polarization_idx,
            :,
            :,
        ]
    )
    grid_min = np.min(
        grid[
            time,
            freq_offset : freq_offset + n_subplots,
            antenna,
            polarization_idx,
            :,
            :,
        ]
    )
    grid_max = np.max(
        grid[
            time,
            freq_offset : freq_offset + n_subplots,
            antenna,
            polarization_idx,
            :,
            :,
        ]
    )

    min_val = np.min(np.array([kl_min, vor_min, grid_min]))
    max_val = np.max(np.array([kl_max, vor_max, grid_max]))

    return min_val, max_val


if __name__ == "__main__":
    # STEP 1
    # Load input/outputs

    # SCREEN FITTING INPUTS:
    # Load solutions.h5 and skymodel
    f = h5py.File("solutions_fullsize.h5", "r")

    # SCREEN FITTING OUTPUTS:
    # The screen fitting library produces as output a .fits image cube.
    # The outputs of the two different algorithms are loaded below as
    # "kl_cube" and "voronoi_cube"
    # the cube dimensions are = ["time", "freqs", "antennas", "pol", "x_coord", "y_coord"]
    kl_cube = fits.open("bigger_screen/kl_0.fits")[0]
    voronoi_cube = fits.open("bigger_screen/tessellated_0.fits")[0]

    # STEP 2
    # Convert the coordinates of the patches in the skymodel to
    # x,y coordinates in the screen
    # build image cube with reference points
    radec_coord = processing_utils.read_patch_list("skymodel.txt", f, "phase000")
    w = wcs.WCS(kl_cube.header)
    [coord_patch_x, coord_patch_y] = processing_utils.get_patch_coordinates(
        radec_coord, w
    )

    # STEP 3
    # Select only points inside the screen
    # If all patches have coordinates which are inside the screen boundaries, list all
    # Otherwise, exclude from patch_idx the patches which fall ouside the screen boundaries
    print(coord_patch_x)
    print(coord_patch_y)
    patch_idx = np.array([0, 1, 2, 3, 4, 5, 6])

    # STEP 4
    # Read amplitude and phase from solution.h5, and subtract reference phase

    ampl = f["sol000/amplitude000/val"]  # [1, 9, 3, 7, 2]
    phase = f["sol000/phase000/val"]  # [3, 9, 3 ,7]

    ntime = phase.shape[0]
    nfreq = phase.shape[1]
    nantennas = phase.shape[2]
    nsources = phase.shape[3]

    ref_antenna = 0

    # re-arrange axes to allow correct broadcasting
    phase_corrected = np.zeros((nantennas, ntime, nfreq, nsources))
    phase_corrected = np.transpose(phase, (2, 0, 1, 3)) - phase[:, :, ref_antenna, :]
    phase_corrected = np.transpose(phase_corrected, (1, 2, 0, 3))

    # STEP 5
    # Build an image cube containing the input values (from the skymodel)
    # at the right xy coordinates, and zeroes elsewhere.
    # A mask is also created which indicates where in the screen the reference points should be placed

    # get the number of pixels in the image
    im_size = kl_cube.data.shape[4]
    npol = 4
    [grid_kl, grid_voronoi, patch_mask] = build_input_grid(
        ntime, nfreq, nantennas, npol, im_size, patch_idx, coord_patch_x, coord_patch_y
    )

    # STEP 6
    # Make plots

    # Select the indexes to plot
    polarization_idx = 1
    time = 0  # there are 3 time slots
    antenna = 1
    freq_offset = 0
    n_subplots = 8
    [colormap_min_val, colormap_max_val] = get_boundaries(
        kl_cube,
        voronoi_cube,
        grid_kl,
        time,
        freq_offset,
        n_subplots,
        antenna,
        polarization_idx,
    )

    plt.figure()
    fig, axarr = plt.subplots(3, n_subplots)
    fig.set_size_inches(60, 20)

    for i in range(n_subplots):
        kl_screen = kl_cube.data[time, i + freq_offset, antenna, polarization_idx, :, :]
        voronoi_screen = voronoi_cube.data[
            time, i + freq_offset, antenna, polarization_idx, :, :
        ]
        ref_screen_kl = grid_kl[time, i + freq_offset, antenna, polarization_idx, :, :]
        ref_screen_voronoi = grid_voronoi[
            time, i + freq_offset, antenna, polarization_idx, :, :
        ]

        kl_screen_with_ref = kl_screen * (~patch_mask) + ref_screen_kl * patch_mask
        voronoi_screen_with_ref = (
            voronoi_screen * (~patch_mask) + ref_screen_voronoi * patch_mask
        )

        axarr[0, i].imshow(
            kl_screen_with_ref, vmin=colormap_min_val, vmax=colormap_max_val
        )
        axarr[1, i].imshow(
            voronoi_screen_with_ref, vmin=colormap_min_val, vmax=colormap_max_val
        )
        axarr[2, i].imshow(ref_screen_kl, vmin=colormap_min_val, vmax=colormap_max_val)
