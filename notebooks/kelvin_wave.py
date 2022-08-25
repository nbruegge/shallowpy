# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import xarray as xr                                                                  
import numpy as np                                                                   
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys                                                                           
import os                                                                            
import glob                                                                          
import pyicon as pyic 
from IPython.display import HTML

# ## Initialize the model

# Initialize default parameters                                                      
# -----------------------------                                                      
exec(open('../shallowpy_defaults.py').read()) 

# +
# Modify default parameters
# -------------------------
# run = __file__.split('/')[-1][:-3]
run = 'jupyter_test'
path_data = f'/Users/nbruegge/work/movies/shallow_py/{run}/'                         
                                                                                     
nx = 100                                                                             
ny = 100                                                                             
nt = 2000                                                                            
#nt = 500                                                                            
#nt = 1                                                                              
                                                                                     
picture_frequency = 0                                                                
output_frequency = 20                                                                
diagnostic_frequency = output_frequency                                              
                                                                                     
dx = 10e3                                                                            
dy = dx                                                                              
#dt = 360.                                                                           
                                                                                     
#grav = 9.81                                                                         
grav = 0.02                                                                          
rho = np.array([1024.])                                                              
nz = rho.size                                                                        
                                                                                     
H0 = 100.                                                                            
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
do_momentum_coriolis_imp = True
do_momentum_pressure_gradient = True
do_height_diffusion = False
# -

# Initialize the grid and initial conditions
# ------------------------------------------
exec(open('../shallowpy_grid_setup.py').read())

# +
# Modify initial conditions
# -------------------------
eta0[0,:,:] = 0.1*np.exp(-((Xt-0.5*Lx)**2+(Yt-Ly)**2)/(1.e-3*(Lx**2+Ly**2)))
eta0[1,:,:] = -H0

ho0 = eta0[:-1,:,:]-eta0[1:,:,:]

maskt0[:,0,:] = 0.
maskt0[:,-1,:] = 0.
maskt0[:,:,0] = 0.
maskt0[:,:,-1] = 0.

ix = np.array([nx//2])
iy = np.array([1*ny//4])
# -

# ## Run the model

# Run the model
# -------------
exec(open('../shallowpy_main.py').read())

# ## Post-process the result

# Do post-processing
# ------------------
exec(open('../pp_main.py').read())

# %%time
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

# ## Some default plots

exec(open('../plot_timeseries.py').read())

exec(open('../plot_fields_tendencies.py').read())

exec(open('../plot_final_initial.py').read())

# ## Make an animation

# +
path_fig = f'{path_data}/anim_01/'
fname_prf = run
fpath = f'{path_data}/test_combined.nc'

mfdset_kwargs = dict(combine='nested', concat_dim='time',
                     data_vars='minimal', coords='minimal', compat='override', join='override',
                    )
ds = xr.open_mfdataset(fpath, **mfdset_kwargs)
#ds = ds.compute()

# +
# prepare the animation

nt = ds.time.size
var = 'ho'
iz = 0
ll=10

hca, hcb = pyic.arrange_axes(1,1, plot_cb=True, asp=1., fig_size_fac=1.5, axlab_kw=None)
ii=-1
fig = plt.gcf()

ii+=1; ax=hca[ii]; cax=hcb[ii]
data = ds['ho'][ll,iz,:,:].compute()
clim = [0, 1e-2]
# clim = 'auto'
hm = pyic.shade(ds.xt/1e3, ds.yt/1e3, data-H0, ax=ax, cax=cax, clim=clim)
ax.set_title('h [m]')
ht = ax.set_title(f'{ds.time[ll].data/60.:.1f}min', loc='right')

#pyic.shade(ds.xt/1e3, ds.yt/1e3, (ds.ho[10,0,:,:]-H0).transpose(), ax=ax, cax=cax, clim='sym')

for ax in hca:
  ax.set_xlabel('x [km]')
  ax.set_ylabel('y [km]')


# -

# function for updating the animation
def run(ll):
    print(f'll = {ll} / {ds.time.size}', end='\r')
    data = ds[var][ll,iz,:,:].data - H0
    hm[0].set_array(data.flatten())
    ht.set_text(f'{ds.time[ll].data/86400.:.1f}days')


# %%time
# --- save the animation
ani = animation.FuncAnimation(fig, run, ds.time.size)
if not os.path.exists(path_fig):
  os.mkdir(path_fig)
ani.save('test.mp4', writer='ffmpeg', fps=40)

# ## Showing the animation

# %%time
HTML(ani.to_jshtml())

# +
#from IPython.display import Video

# +
#from IPython.display import HTML

# +
# #!ln -sf "~/work/movies/shallow_py/examp_kelvin_wave/anim_01/kelvin_wave.mp4" .

# +
#Video("/Users/nbruegge/work/movies/shallow_py/examp_kelvin_wave/anim_01/OUT.mp4", embed=True)
# Video("/Users/nbruegge/work/movies/shallow_py/examp_kelvin_wave/anim_01/kelvin_wave.mp4", embed=True)
# -


