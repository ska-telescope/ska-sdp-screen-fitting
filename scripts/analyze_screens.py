"""Script to plot screens with input solutions overlay"""
from astropy import wcs
import numpy as np
import sys

sys.path.insert(1, "../src/ska_sdp_screen_fitting/utils")
import processing_utils


def get_boundaries(
    kl_cube,
    voronoi_cube,
    values_kl,
    values_vornoi,
    time,
    freq_offset,
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

    s = np.s_[time, freq_offset, antenna, polarization_idx]

    kl_min = np.min(kl_cube.data[s])
    kl_max = np.max(kl_cube.data[s])
    vor_min = np.min(voronoi_cube.data[s])
    vor_max = np.max(voronoi_cube.data[s])
    values_kl_min = np.min(values_kl[s])
    values_kl_max = np.max(values_kl[s])
    values_vornoi_min = np.min(values_vornoi[s])
    values_vornoi_max = np.max(values_vornoi[s])

    min_val = np.min([kl_min, vor_min, values_kl_min, values_vornoi_min])
    max_val = np.max([kl_max, vor_max, values_kl_max, values_vornoi_max])

    return min_val, max_val


def get_phase_corrected(phase, ref_antenna=0):
    """Subtract reference phase
    Parameters
    ----------
    phase : ndarray
        Input phases as read from the h5 solutions file
    ref_antenna : int
        Antenna to be considered as phase reference
   
    Returns
    -------
    phase_corrected : ndarray
        Phase corrected for the reference
    """

    ntime = phase.shape[0]
    nfreq = phase.shape[1]
    nantennas = phase.shape[2]
    nsources = phase.shape[3]

    # re-arrange axes to allow correct broadcasting
    phase_corrected = np.zeros((nantennas, ntime, nfreq, nsources))
    phase_corrected = np.transpose(phase, (2, 0, 1, 3)) - phase[:, :, ref_antenna, :]
    phase_corrected = np.transpose(phase_corrected, (1, 2, 0, 3))

    return phase_corrected


if __name__ == "__main__":
    # STEP 1
    # Load input/outputs
    # SCREEN FITTING INPUTS:
    # Load solutions.h5 and skymodel
    f = h5py.File("../resources/solutions.h5", "r")

    # SCREEN FITTING OUTPUTS:
    # The screen fitting library produces as output a .fits image cube.
    # The outputs of the two different algorithms are loaded below as
    # "kl_cube" and "voronoi_cube"
    # the cube dimensions are = ["time", "freqs", "antennas", "pol", "x_coord", "y_coord"]
    kl_cube = fits.open("../resources/kl_0.fits")[0]
    voronoi_cube = fits.open("../resources/tessellated_0.fits")[0]

    # STEP 2
    # Convert the coordinates of the patches in the skymodel to
    # x,y coordinates in the screen
    # build image cube with reference points
    radec_coord = processing_utils.read_patch_list(
        "../resources/skymodel.txt", f, "phase000"
    )
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
    phase = f["sol000/phase000/val"]  # [1, 9, 3, 7, 2]
    phase_corrected = get_phase_corrected(phase)

    # option 1
    # If available in the h5 solution file, read amplitudes from there
    # Expand dimensions on time axis of amplitude
    # ampl2 = np.zeros([3] + list(ampl.shape[1:]))
    # ampl2[:] = ampl

    # option 2
    # Dummy values are added here, as amplitudes are not available
    # The shape of the amplitude array is the same as phases, with
    # one extra dimension to specify XX and YY polarization
    ampl = np.ones((phase.shape[0], phase.shape[1], phase.shape[2], phase.shape[3], 2))

    # STEP 5
    # Calculate the screen values at the patches position
    polarization_idx = 3  # choose {0, 1, 2, 3}
    funcs = [np.cos, np.sin]
    val = (
        funcs[polarization_idx % 2](phase_corrected)
        * ampl[:, :, :, :, polarization_idx // 2]
    )

    # STEP 6
    # Make plots
    time = 0
    antenna = 1
    freq_offset = 3

    [colormap_min_val, colormap_max_val] = get_boundaries(
        kl_cube, voronoi_cube, val, val, time, freq_offset, antenna, polarization_idx
    )

    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(
        kl_cube.data[time, freq_offset, antenna, polarization_idx, :, :],
        vmin=colormap_min_val,
        vmax=colormap_max_val,
    )
    ax[0].scatter(
        coord_patch_x,
        coord_patch_y,
        c=val[time, freq_offset, antenna, :],
        vmin=colormap_min_val,
        vmax=colormap_max_val,
        edgecolors="black",
        linewidth=0.5,
        s=150,
    )
    ax[0].set_title("Karhunen Loève screens")
    ax[1].imshow(
        voronoi_cube.data[time, freq_offset, antenna, polarization_idx, :, :],
        vmin=colormap_min_val,
        vmax=colormap_max_val,
    )
    ax[1].scatter(
        coord_patch_x,
        coord_patch_y,
        c=val[time, freq_offset, antenna, :],
        vmin=colormap_min_val,
        vmax=colormap_max_val,
        edgecolors="black",
        linewidth=0.5,
        s=150,
    )
    ax[1].set_title("Voronoi screens")
