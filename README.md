# Sequential electroreductions at alternating polarity are more thermodynamically wasteful and less selective than at direct currents
This is a repository of code accompanying the research article

### Eric S. Larsen, Louis P. J.-L. Gadina, Yaroslav I. Sobolev, Rafal Frydrych, Kisung Lee, Jonathan Sabate Del-Rio, Bartosz A. Grzybowski

Experimental data and results of numerical calculations are too large (~29 Gb) for Github and are available from separate repository in Harvard Dataverse: 

## Installation
This code is compatible with Python 3.11.7

The primary Python dependencies are:
matplotlib 3.8.0
emcee 3.1.6
NumPy 1.26.4
SciPy 1.11.4
Other Python dependencies are standard -- come pre-installed with Anaconda distribution.

## Contents
The 4 pythons files are 
convert_ascii_to_binary.py”, converts recorded date from CSV to NPY.
2.	“Gadina_Sobolev_fit.py”, handles the evaluation of current pulses given voltage, the frequency of acquisition of the recording and the equivalent circuit parameters.
3.	“load_backend.py”, reads the emcee results and generates the corresponding corner plots of “.h5” files.
4.	“automated_pulse_analyser.py”, performs the pre-processing and grouping of pulses (described below) of current and voltage recordings, runs EMCEE and handles the respective plotting, and fits the current curves using the curve_fit method from scipy.optimize and the “Gadina_Sobolev_fit.py” module.
