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
run = __file__.split('/')[-1][:-3]
path_data = f'/Users/nbruegge/work/movies/shallow_py/{run}/'

nx = 100
ny = 100
nt = 2000
nt = 500
#nt = 1

picture_frequency = 0
output_frequency = 20
diagnostic_frequency = output_frequency

dx = 10e3
dy = dx
#dt = 360.

grav = 9.81
rho = np.array([1024.])
nz = rho.size

H0 = 1000.
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
beta = 0.*1e-11

do_momentum_advection = False                   
do_momentum_diffusion = False
do_momentum_coriolis_exp = False
do_momentum_coriolis_imp = False
do_momentum_pressure_gradient = True
do_height_diffusion = False

# Initialize the grid and initial conditions
# ------------------------------------------
exec(open('./shallowpy_grid_setup.py').read())

# Modify initial conditions
# -------------------------
#ho0 = 0.01*(Xt-Lx/2.)/Lx
#ho0 = H0+0.1*np.sin(Xt/(Lx+dx)*2*np.pi*2)
eta0[0,:,:] = 0.1*np.exp(-((Xt-0.5*Lx)**2)/(1.e-3*(Lx**2+Ly**2)))
eta0[1,:ny//2,:] = -H0
eta0[1,ny//2:,:] = -H0/2.
#ho0[:ny//2,:] += H0
#ho0[ny//2:,:] += H0/2.
#ho0 = ho0[np.newaxis,:,:]
#H0 = 0.
#ho0 += H0
ho0 = eta0[:-1,:,:]-eta0[1:,:,:]

#uo0 = cph + 0.*Xu
#uo0 = uo0[np.newaxis,:,:]

maskt0[:,0,:] = 0.
maskt0[:,-1,:] = 0.
maskt0[:,ny//2-1:ny//2+1,:] = 0.

#eta_bot0[:,:ny//2,:] = H0
#eta_bot0[:,ny//2:,:] = H0
#H0 = 0.

#ix = np.array([nx//2, nx//2])
#iy = np.array([1*ny//4, 3*ny//4])
ix = np.array([nx//2])
iy = np.array([1*ny//4])

# Run the model
# -------------
exec(open('./shallowpy_main.py').read())

# Do post-processing
# ------------------
exec(open('./pp_main.py').read())

# Combine all netcdf files
# ------------------------
#if output_frequency>0:
if True:
  mfdset_kwargs = dict(combine='nested', concat_dim='time',
                       data_vars='minimal', coords='minimal', compat='override', join='override',
                      )
  ds = xr.open_mfdataset(f'{path_data}/{file_prfx}_????.nc', **mfdset_kwargs)
  fpath = f'{path_data}/{file_prfx}_combined.nc'
  print(f'Save file {fpath}')
  ds['ho'] = ds.ho.where(masktp==1)
  ds['uo'] = ds.ho.where(maskup==1)
  ds['vo'] = ds.ho.where(maskvp==1)
  ds = ds.compute()
  ds['ho'][0,:,:,:] = np.nan
  ds.to_netcdf(fpath)


# Visualize results
# -----------------
plt.close('all')

exec(open('./plot_timeseries.py').read())

exec(open('./plot_fields_tendencies.py').read())

exec(open('./plot_final_initial.py').read())

# ---
hca, hcb = pyic.arrange_axes(2,1, plot_cb=False, asp=1., fig_size_fac=1.5)
ii=-1

iy = 3*ny//4

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

# ---
hca, hcb = pyic.arrange_axes(2,1, plot_cb=True, asp=1., fig_size_fac=1.5)
ii=-1

ii+=1; ax=hca[ii]; cax=hcb[ii]
iy = 1*ny//4
pyic.shade(ds.time, xt/1e3, (ds.ho[:,0,iy,:]-H0).transpose(), ax=ax, cax=cax, clim='sym')
xgrav = np.sqrt(grav*H0)*ds.time + Lx/2.
ax.plot(ds.time, xgrav/1e3, color='k', label='\sqrt{g H}')
xgrav = -np.sqrt(grav*H0)*ds.time + Lx/2.
ax.plot(ds.time, xgrav/1e3, color='k')
ax.set_title(f'ho with H0 = {H0}m')

ii+=1; ax=hca[ii]; cax=hcb[ii]
iy = 3*ny//4
pyic.shade(ds.time, xt/1e3, (ds.ho[:,0,iy,:]-H0/2.).transpose(), ax=ax, cax=cax, clim='sym')
xgrav = np.sqrt(grav*H0/2.)*ds.time + Lx/2.
ax.plot(ds.time, xgrav/1e3, color='k', label='\sqrt{g H}')
xgrav = -np.sqrt(grav*H0/2.)*ds.time + Lx/2.
ax.plot(ds.time, xgrav/1e3, color='k')
ax.set_title(f'ho with H0 = {H0/2}m')

for ax in hca:
  ax.legend()
  ax.set_xlabel('time [sec]')
  ax.set_ylabel('x [km]')

plt.show()
