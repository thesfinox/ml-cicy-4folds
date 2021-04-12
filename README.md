# AI for CICY 4-folds

## Use on the cluster

1. Use `preprocessing` to create the training sets and the grid of hyperparameters,
2. Run `cicy` to run the optimisation.

Use `clean` to remove error and output files.

## Standalone use

1. Use `python sets.py -h` to show the command line parameters for the creation of the datasets,
1. Use `python grid.py -h` to show the command line parameters for the creation of the hyperparameter list,
2. Run `python cicy.py -h` for a list of parameters for the analysis.

## Virtual Environment

The file `requirements.yml` contains the packages needed to run the analysis.
Using **conda**, it is possible to replicate the environment via `conda env create -f requirements.yml`.

