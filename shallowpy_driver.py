import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import argparse

help_text = """Driver for shallowpy.

!!! This was experimental but should not be used. !!!
"""

# --- read input arguments
parser = argparse.ArgumentParser(description=help_text, formatter_class=argparse.RawTextHelpFormatter)

# --- necessary arguments
parser.add_argument('fpath_config', metavar='fpath_config', type=str,
                    help='path to quickplot configure file')
# --- optional arguments
iopts = parser.parse_args()
fpath_config = iopts.fpath_config

# Initialize default parameters
# -----------------------------
exec(open('./shallowpy_defaults.py').read())

# Modify default parameters
# -------------------------
mode = 'default_parameters'
exec(open(fpath_config).read())

# Initialize the grid and initial conditions
# ------------------------------------------
exec(open('./shallowpy_grid_setup.py').read())

# Modify initial conditions
# -------------------------
mode = 'initial_conditions'
exec(open(fpath_config).read())

# Run the model
# -------------
exec(open('./shallowpy_main.py').read())

# Visualize results
# -----------------
