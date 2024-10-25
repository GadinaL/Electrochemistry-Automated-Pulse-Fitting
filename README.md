# Sequential electroreductions at alternating polarity are more thermodynamically wasteful and less selective than at direct current
This is a repository of code accompanying the research article

### Eric S. Larsen, Louis P. J.-L. Gadina, Yaroslav I. Sobolev, Jonathan Sabate Del-Rio, Rafal Frydrych, Kisung Lee, Bartosz A. Grzybowski

Experimental data and results of numerical calculations are too large (~29 Gb) for Github and are available from separate repository in Harvard Dataverse: 

## Installation
This code is compatible with Python 3.11.7

The primary Python dependencies are:

- matplotlib 3.8.0,
- emcee 3.1.6,
- NumPy 1.26.4,
- SciPy 1.11.4.
  
Other Python dependencies are standard -- come pre-installed with Anaconda distribution.

## Contents
 
1.	`convert_ascii_to_binary.py`, converts recorded date from CSV to NPY.
2.	`Gadina_Sobolev_fit.py`, handles the evaluation of current pulses given voltage, the frequency of acquisition of the recording and the equivalent circuit parameters.
3.	`load_backend.py`, reads the emcee results and generates the corresponding corner plots of `.h5` files.
4.	`automated_pulse_analyser.py`, performs the pre-processing and grouping of pulses (described below) of current and voltage recordings, runs EMCEE and handles the respective plotting, and fits the current curves using the curve_fit method from scipy.optimize and the `Gadina_Sobolev_fit.py` module.
5.	`simulate_kinetic_shapes.py` contains methods for fitting kinetic model to the yield-vs-time curves. Execute to reproduce Figures 7c and Figure S35a-e from the accompanying research article.
6.	`simulate_kinetic_shapes_pulselengths.py` uses methods from `simulate_kinetic_shapes.py` and analyzes the dependence of kinetic rate constants on the length of the square pulse. Also fits kinetic model to DC data. Execute to reproduce Figures 6a-c and Figure S35f from the accompanying research article.
7.	`error_analysis.py` constructs linear model of the instrumental errors. Run to reproduce Figure S37.

To reproduce the Figure S36, uncomment the [line 156 in ``simulate_kinetic_shapes.py``](https://github.com/GadinaL/Automated_Pulse_Fitting-Electrochemistry/blob/2ad77a00de21ce842dd79ca302897557b8c6a54c/simulate_kinetics_shapes.py#L156) and run `simulate_kinetic_shapes.py` and `simulate_kinetic_shapes_pulselengths.py`.

Yield curves are in the folder `data/yield_curves`.
Additonnaly, EIS data used in the research article are also available in `EIS Data` folder.
