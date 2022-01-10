import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import pyicon as pyic
import argparse

# After Early et al. 2011, JPO

# Initialize default parameters
# -----------------------------
exec(open('./shallowpy_defaults.py').read())

# Modify default parameters
# -------------------------
run = __file__.split('/')[-1][:-3]
path_data = f'/Users/nbruegge/work/movies/shallow_py/{run}/'

fac = 1
nx = 200*fac
ny = 120*fac
nt = 8000*fac
#nt = 2

picture_frequency = 0
output_frequency = 50
diagnostic_frequency = 50

x0 = -1500e3
y0 = -600e3
dx = 10e3
dy = dx
#dt = 360.

#grav = 9.81
grav = 0.01
rho = np.array([1024.])
nz = rho.size

H0 = 800.
cph = np.sqrt(grav*H0)
dist = dt*nt * cph
dt  = 0.1*dx/np.sqrt(grav*H0)
#dt = 360.

nspx = 1
nspy = 1
epsab = 0.01

kh = 1000.
Ah = kh

f0 = 1e-4
beta = 1e-11
R_earth = 6371e3
f0 = 2 * 2*np.pi/86400*np.sin(24.*np.pi/180.)
beta = 2 * 2*np.pi/86400/R_earth * np.cos(24.*np.pi/180.)
#Y0 = ny*dy/2.

Lr = np.sqrt(grav*H0)/f0
c_grav = np.sqrt(grav*H0)
c_ross = -beta*Lr**2 

# c_grav/c_ross = f0*np.sqrt(grav*H0) / (beta*grav*H0) = f0/beta / np.sqrt(grav*H0)

do_momentum_advection = False                   
do_momentum_diffusion = False
do_momentum_coriolis_exp = False
do_momentum_coriolis_imp = True
do_momentum_pressure_gradient = True
do_height_diffusion = False
do_height_advection = True
do_linear_height_advection = True

# Initialize the grid and initial conditions
# ------------------------------------------
exec(open('./shallowpy_grid_setup.py').read())

# Modify initial conditions
# -------------------------
#ho0 = 0.01*(Xt-Lx/2.)/Lx
#ho0 = H0+0.1*np.sin(Xt/(Lx+dx)*2*np.pi*2)
#L2 = 1.e-3*(Lx**2+Ly**2)
L = 80e3
#ho0 = H0+0.15*np.exp(-((Xt-0.75*Lx)**2+(Yt-0.5*Ly)**2)/L**2)
ho0 = H0+0.15*np.exp(-(Xt**2+Yt**2)/L**2)
ho0 = ho0[np.newaxis,:,:]
#H0 = 0.
#ho0 += H0

#uo0 = cph + 0.*Xu
#uo0 = uo0[np.newaxis,:,:]

#maskt0[:,:,0] = 0.
#maskt0[:,:,-1] = 0.

ix = np.array([nx//2])
iy = np.array([ny//2])

# Run the model
# -------------
exec(open('./shallowpy_main.py').read())

# Combine all netcdf files
# ------------------------
#if output_frequency>0:
if False:
  mfdset_kwargs = dict(combine='nested', concat_dim='time',
                       data_vars='minimal', coords='minimal', compat='override', join='override',
                      )
  ds = xr.open_mfdataset(f'{path_data}/{file_prfx}_????.nc', **mfdset_kwargs)
  fpath = f'{path_data}/{file_prfx}_combined.nc'
  print(f'Save file {fpath}')
  ds.to_netcdf(fpath)

# Visualize results
# -----------------
plt.close('all')

exec(open('./plot_timeseries.py').read())

exec(open('./plot_fields_tendencies.py').read())

exec(open('./plot_final_initial.py').read())

# ---
hca, hcb = pyic.arrange_axes(2,1, plot_cb=True, asp=1., fig_size_fac=1.5)
ii=-1

iy = ny//2

ii+=1; ax=hca[ii]; cax=hcb[ii]
ax.plot(xt/1e3, uo0[0,iy,:], label='uo0')
ax.plot(xt/1e3, uop[0,iy,:], label='uop')
ax.plot(xt/1e3, vo0[0,iy,:], label='vo0')
ax.plot(xt/1e3, vop[0,iy,:], label='vop')
ax.legend()

ii+=1; ax=hca[ii]; cax=hcb[ii]
ax.plot(xt/1e3, ho0[0,iy,:], label='ho0')
ax.plot(xt/1e3, hop[0,iy,:], label='hop')
ax.legend()


plt.show()
