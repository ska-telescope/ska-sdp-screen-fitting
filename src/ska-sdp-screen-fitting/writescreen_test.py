import astropy.io.fits as fits
import numpy as np
import pyrap.tables as pt
import tables
from astropy import wcs

do_plot = False
# myt=tables.open_file("data/NsolutionsDDE_2.5Jy_tecandphasePF_correctedlosoto_july2018.h5")
myt = tables.open_file(
    "data/solutionsDDE_2.5Jy_tecandphasePF_correctedlosoto_fulltime_dec27.h5"
)
tec = myt.root.sol000.tec000.val[:]
ph0 = myt.root.sol000.scalarphase000.val[:]
freq = pt.table("data/cP126+65BEAM_1_chan200-210.ms/SPECTRAL_WINDOW").getcol(
    "CHAN_FREQ"
)


def create_Aterms(posx, posy, order=3):
    Aterms = []
    for x in range(order):
        tmpterms = [posx ** x]
        for y in range(x + 1, order):
            tmpterms.append(tmpterms[-1] * posy)
        Aterms += tmpterms
    return Aterms


phase = ph0 - 8.4479745e9 * tec / freq


src = myt.root.sol000.source.cols.dir[:]
name = myt.root.sol000.source.cols.name[:]
name2 = myt.root.sol000.tec000.dir[:]
srcpos = np.array([src[list(name).index(i)] for i in name2])
phase = np.remainder(phase + np.pi, 2 * np.pi) - np.pi
avgphase = np.angle(np.sum(np.exp(1.0j * phase), axis=0))
phase = (
    np.remainder(phase - avgphase[np.newaxis] + np.pi, 2 * np.pi)
    - np.pi
    + avgphase[np.newaxis]
)

inv_var = 1.0 / np.var(phase[:, 1:], axis=(0, 1, 3))

Atec = -freq[0] / 8.4479745e9
# sol_tec=np.dot((1./np.dot(Atec,Atec)),np.dot(Atec,fitted_data.transpose((0,1,3,2)))).transpose((0,2,1))
# sol_tec=Atec[np.newaxis,np.newaxis,np.newaxis]*fitted_data
sol_tec = Atec[np.newaxis, np.newaxis, np.newaxis] * phase
time = myt.root.sol000.tec000.time[:]
header = fits.getheader("data/images/nobeam-image.fits")
nrpix = 3
dtime = 5
dl = header["CDELT1"]
dm = header["CDELT2"]
Nl = header["NAXIS1"]
Nm = header["NAXIS2"]
dl *= Nl / nrpix
dm *= Nm / nrpix

for i in header.keys():
    if i[-1] == "3":
        header[i.replace("3", "4")] = header[i]
header["NAXIS"] = 5
header["CTYPE3"] = "ANTENNA"
header["CRVAL3"] = 1
header["CRPIX3"] = 1
header["CDELT3"] = 1
header["NAXIS3"] = phase.shape[1]
header["CTYPE5"] = "TIME"
header["CRPIX5"] = 1
header["CDELT5"] = dtime * (time[1] - time[0])
header["CRVAL5"] = time[0]
header["NAXIS5"] = phase.shape[0] / dtime
header["NAXIS1"] = nrpix
header["NAXIS2"] = nrpix
header["CRPIX1"] = int(nrpix / 2) + 1
header["CRPIX2"] = int(nrpix / 2) + 1
header["CDELT1"] = dl
header["CDELT2"] = dm

w = wcs.WCS(fits.getheader("data/images/nobeam-image.fits"))
x = np.linspace(0, Nl, nrpix, False)
y = np.linspace(0, Nm, nrpix, False)
xy = np.meshgrid(x, y)
coords = w.wcs_pix2world(xy[0], xy[1], 0, 0, 1)
coords[0] = np.radians(coords[0])
coords[1] = np.radians(coords[1])
print(np.degrees(coords[0]) * 12 / 180.0, np.degrees(coords[1]))

b = np.zeros(
    ((phase.shape[0] / dtime,) + phase.shape[1:2] + (1, nrpix, nrpix))
)
for itm in range(0, 3595, dtime):
    print(itm)
    b[
        itm / dtime : itm / dtime + 1,
        :,
        0,
        : nrpix / 2,
        nrpix / 2 + nrpix % 2 :,
    ] = np.average(sol_tec[itm : itm + dtime, :, 0, 5], axis=0)[
        :, np.newaxis, np.newaxis
    ]

if do_plot:
    from pylab import cla, imshow, pause, scatter, title

    for itm in range(100):
        cla()
        imshow(
            b[itm, 61],
            interpolation="nearest",
            vmin=-0.1,
            vmax=0.1,
            cmap="hsv",
            origin="lower",
            extent=[
                coords[0][0, 0],
                coords[0][-1, -1],
                coords[1][0, 0],
                coords[1][-1, -1],
            ],
        )
        scatter(
            srcpos[:, 0],
            srcpos[:, 1],
            c=sol_tec[itm * dtime, 61, :, 5],
            s=150,
            edgecolors="k",
            vmin=-0.1,
            vmax=0.1,
            cmap="hsv",
        )
        title(str(itm))
        pause(0.01)
b = b.transpose((0, 2, 1, 4, 3))  # axes are reversed in fits?
fits.writeto("data/testscreen_3pix_tm5.fits", data=b, header=header)
