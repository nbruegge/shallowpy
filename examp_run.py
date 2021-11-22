import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import pyicon as pyic
import argparse

# Initialize default parameters
# -----------------------------
exec(open('./shallowpy_defaults.py').read())

# Modify default parameters
# -------------------------
nx = 120
ny = 120
nt = 500
#nt = 20

picture_frequency = 0
output_frequency = 0

dx = 10e3
dy = dx
dt = 720.

grav = 9.81
rho = np.array([1024.])
nz = rho.size

nspx = nspy = 1
epsab = 0.01

kh = 20.
Ah = kh

f0 = 1e-4
beta = 0.*1e-11

# Initialize the grid and initial conditions
# ------------------------------------------
exec(open('./shallowpy_grid_setup.py').read())

# Modify initial conditions
# -------------------------
ho0 = 0.01*np.exp(-((Xt-0.5*Lx)**2+(Yt-0.5*Ly)**2)/(1.e-3*(Lx**2+Ly**2)))

ho0 = ho0[np.newaxis,:,:]
H0 = 0.
ho0 += H0

# Run the model
# -------------
exec(open('./shallowpy_main.py').read())

### Combine all netcdf files
### ------------------------
##mfdset_kwargs = dict(combine='nested', concat_dim='time',
##                     data_vars='minimal', coords='minimal', compat='override', join='override',
##                    )
##ds = xr.open_mfdataset(f'{path_data}/*.nc', **mfdset_kwargs)
##fpath = f'{path_data}/{file_prfx}_combined.nc'
##print(f'Save file {fpath}')
##ds.to_netcdf(fpath)

# Visualize results
# -----------------
plt.close('all')

exec(open('./plot_timeseries.py').read())

exec(open('./plot_fields_tendencies.py').read())

exec(open('./plot_final_initial.py').read())

plt.show()
