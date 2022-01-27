# Screen Fitting

## Description
This module can be used to convert a direction dependent calibration file (in h5 format) into a set of screens (image cube in fits format).
The input calibration file has one solution per antenna, frequency and solution interval; the output has instead one screen per antenna, frequency and solution interval. 
There are two algotihms available for screen fitting:
- Voronoi
- Karhunen Loeve

The screens obtained can be used to apply direction dependent effects during the imaging process with IDG in WSClean. 
Refer to https://wsclean.readthedocs.io/en/latest/a_term_correction.html?highlight=screen in the section "Diagonal gain correction" for more details.

## Badges
TODO

## Visuals
The gif below shows the output of the KL (left) and Voronoi (right) screen-fitting algorithm for a fixed station and solution interval, varying in frequency. 

<img src="resources/kl_screen_fitting.gif" width="400" height="400" />
<img src="resources/voronoi_screen_fitting.gif" width="400" height="400" />

## Installation
TODO

## Usage 
TODO

## Authors and acknowledgment
TODO

## License
TODO