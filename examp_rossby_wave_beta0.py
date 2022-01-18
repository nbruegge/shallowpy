import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import pyicon as pyic
import argparse
import shutil

# After Early et al. 2011, JPO

# Initialize default parameters
# -----------------------------
exec(open('./shallowpy_defaults.py').read())

mode = sys.argv[1]
#iid = int(sys.argv[2])
iRe = sys.argv[2]
init_with_geostrophic_adjustment = False
do_upwind_advection = False

if sys.argv[1]=='linear':
  do_linear = True
elif sys.argv[1]=='nonlinear':
  do_linear = False
else:
  print(f'Wrong mode \'{sys.argv[1]}\'')
  sys.exit()
if init_with_geostrophic_adjustment:
  adj_str = '_gadj'
else:
  adj_str = ''
if do_upwind_advection:
  upw_str = ''
else:
  upw_str = '_adv2nd'

# Modify default parameters
# -------------------------
run = __file__.split('/')[-1][:-3]
if do_linear:
  #run = f'{run}_linear_{iid:02d}'
  run = f'{run}_linear{adj_str}{upw_str}_iRe{iRe}'
else:
  #run = f'{run}_nonlinear_{iid:02d}'
  run = f'{run}_nonlinear{adj_str}{upw_str}_iRe{iRe}'
#path_data = f'/Users/nbruegge/work/movies/shallow_py/{run}/'
path_data = f'/home/m/m300602/work/shallowpy/{run}/'

print(path_data)

# make directory and copy this script
try:
  os.makedirs(path_data)
except:
  pass
shutil.copyfile(__file__, path_data+__file__.split('/')[-1])

fac = 1
dt_fac = 0.5
nx = 200*fac
ny = 120*fac
#nt = int(180000*fac/dt_fac)
nt = int(1800*fac/dt_fac)
#nt = 10

picture_frequency = 0
output_frequency = 240
diagnostic_frequency = 240

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
#dt  = dt_fac * 0.1*dx/np.sqrt(grav*H0)
dt = dt_fac * 360.

nspx = 1
nspy = 1
epsab = 0.01

U0 = 1e-3
#iRe = 5e-4
iRe = float(iRe)
if do_upwind_advection:
  do_height_diffusion = False # (since we have 1st order upwind adv. scheme)
  kh = 0.
else:
  do_height_diffusion = True
  kh = iRe*U0*dx
Ah = iRe*U0*dx

#f0 = 1e-4
#beta = 1e-11
R_earth = 6371e3
f0 = 2 * 2*np.pi/86400*np.sin(24.*np.pi/180.)
beta = 0 * 2 * 2*np.pi/86400/R_earth * np.cos(24.*np.pi/180.)
#Y0 = ny*dy/2.

Lr = np.sqrt(grav*H0)/f0
c_grav = np.sqrt(grav*H0)
c_ross = -beta*Lr**2 

Eta0 = 100.
N0 = 150.
L0 = 80e3
U0 = grav*Eta0/(f0*Lr)
beta_nli = U0/(beta*Lr**2)

# c_grav/c_ross = f0*np.sqrt(grav*H0) / (beta*grav*H0) = f0/beta / np.sqrt(grav*H0)
#sys.exit()

if do_linear:
  do_linear_height_advection = True
  do_momentum_advection = False
else:
  do_linear_height_advection = False
  do_momentum_advection = True
do_momentum_diffusion = True
do_momentum_coriolis_imp = True
do_momentum_pressure_gradient = True
do_height_advection = True

# Initialize the grid and initial conditions
# ------------------------------------------
exec(open('./shallowpy_grid_setup.py').read())

# Modify initial conditions
# -------------------------
#ho0 = 0.01*(Xt-Lx/2.)/Lx
#ho0 = H0+0.1*np.sin(Xt/(Lx+dx)*2*np.pi*2)
#L2 = 1.e-3*(Lx**2+Ly**2)
ho0 = H0+N0*np.exp(-(Xt**2+Yt**2)/L0**2)
ho0 = ho0[np.newaxis,:,:]
#H0 = 0.
#ho0 += H0

#uo0 = cph + 0.*Xu
#uo0 = uo0[np.newaxis,:,:]
if init_with_geostrophic_adjustment:
  vo0 =  grav/f0 * N0 * (-2*Xt/L0**2) * np.exp(-(Xt**2+Yu**2)/L0**2) 
  uo0 = -grav/f0 * N0 * (-2*Yt/L0**2) * np.exp(-(Xu**2+Yt**2)/L0**2) 
  uo0 = uo0[np.newaxis,:,:]
  vo0 = vo0[np.newaxis,:,:]

maskt0[:,:,0] = 0.
maskt0[:,:,-1] = 0.
maskt0[:,0,:] = 0.
maskt0[:,-1,:] = 0.

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
