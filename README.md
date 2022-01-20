# shallowpy

Shallowpy is shallow water model with full nonlinear capabilities. The model is
thought to be used by student or as preparation for course material.
Therefore, whenever appropriate simplicity is given a stronger weight than
numerical accuracy of the single algorithms.

## Installation

Open file `shallowpy_main.py` and `examp_rossby_wave.py` to see which modules are used by the model. 
Install those e.g. by conda. The python module `pyicon` is not a standard python library but it can 
be downloaded from

https://gitlab.dkrz.de/m300602/pyicon

It is only used for plotting the results. So when the plots are disabled one can use `shallowpy` 
without pyicon.

## Running the model

Copy one of the `examp_*.py` scripts and modify them according to your needs. 
Then run this script in Jupyter Notebook, ipython session or with 
`python examp_my_script.py`.

