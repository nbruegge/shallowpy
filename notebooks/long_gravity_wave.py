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

# + papermill={"duration": 1.1145, "end_time": "2022-08-26T10:18:37.490834", "exception": false, "start_time": "2022-08-26T10:18:36.376334", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# + papermill={"duration": 1.546376, "end_time": "2022-08-26T10:18:39.052443", "exception": false, "start_time": "2022-08-26T10:18:37.506067", "status": "completed"} tags=[]
import xarray as xr                                                                  
import numpy as np                                                                   
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys                                                                           
import os                                                                            
import glob                                                                          
from IPython.display import HTML
sys.path.append('../')
from shallowpy_plotting import arrange_axes, shade

# + [markdown] papermill={"duration": 0.012879, "end_time": "2022-08-26T10:18:39.079348", "exception": false, "start_time": "2022-08-26T10:18:39.066469", "status": "completed"} tags=[]
# ## Initialize the model

# + papermill={"duration": 0.055364, "end_time": "2022-08-26T10:18:39.151745", "exception": false, "start_time": "2022-08-26T10:18:39.096381", "status": "completed"} tags=[]
# Initialize default parameters                                                      
# -----------------------------                                                      
exec(open('../shallowpy_defaults.py').read()) 

# + papermill={"duration": 0.05318, "end_time": "2022-08-26T10:18:39.220099", "exception": false, "start_time": "2022-08-26T10:18:39.166919", "status": "completed"} tags=[]
# Modify default parameters
# -------------------------
run = 'long_gravity_wave'
path_data = f'/Users/nbruegge/work/movies/shallow_py/{run}/'

nx = 100
ny = 100
nt = 500

picture_frequency = 0
output_frequency = 20
diagnostic_frequency = output_frequency

dx = 10e3
dy = dx

grav = 9.81
rho = np.array([1024.])
nz = rho.size

H0 = 1000.
cph = np.sqrt(grav*H0)
dist = dt*nt * cph
dt  = 0.1*dx/np.sqrt(grav*H0)

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

# + papermill={"duration": 0.048357, "end_time": "2022-08-26T10:18:39.282215", "exception": false, "start_time": "2022-08-26T10:18:39.233858", "status": "completed"} tags=[]
# Initialize the grid and initial conditions
# ------------------------------------------
exec(open('../shallowpy_grid_setup.py').read())

# + papermill={"duration": 0.056366, "end_time": "2022-08-26T10:18:39.352256", "exception": false, "start_time": "2022-08-26T10:18:39.295890", "status": "completed"} tags=[]
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

# + [markdown] papermill={"duration": 0.014051, "end_time": "2022-08-26T10:18:39.380689", "exception": false, "start_time": "2022-08-26T10:18:39.366638", "status": "completed"} tags=[]
# ## Run the model

# + papermill={"duration": 2.745671, "end_time": "2022-08-26T10:18:42.141606", "exception": false, "start_time": "2022-08-26T10:18:39.395935", "status": "completed"} tags=[]
# Run the model
# -------------
exec(open('../shallowpy_main.py').read())

# + [markdown] papermill={"duration": 0.015096, "end_time": "2022-08-26T10:18:42.174815", "exception": false, "start_time": "2022-08-26T10:18:42.159719", "status": "completed"} tags=[]
# ## Post-process the result

# + papermill={"duration": 0.058064, "end_time": "2022-08-26T10:18:42.250589", "exception": false, "start_time": "2022-08-26T10:18:42.192525", "status": "completed"} tags=[]
# Do post-processing
# ------------------
exec(open('../pp_main.py').read())

# + papermill={"duration": 0.740798, "end_time": "2022-08-26T10:18:43.006992", "exception": false, "start_time": "2022-08-26T10:18:42.266194", "status": "completed"} tags=[]
# %%time
# Combine all netcdf files
# ------------------------
#if output_frequency>0:
if True:
    mfdset_kwargs = dict(combine='nested', concat_dim='time',
        data_vars='minimal', coords='minimal', compat='override', join='override',
        parallel=True
    )
    ds = xr.open_mfdataset(f'{path_data}/{file_prfx}_????.nc', **mfdset_kwargs)
    fpath = f'{path_data}/{file_prfx}_combined.nc'
    print(f'Save file {fpath}')
    ds['ho'] = ds.ho.where(masktp==1)
    ds['uo'] = ds.uo.where(maskup==1)
    ds['vo'] = ds.vo.where(maskvp==1)
    ds.to_netcdf(fpath)

# + [markdown] papermill={"duration": 0.015921, "end_time": "2022-08-26T10:18:43.039205", "exception": false, "start_time": "2022-08-26T10:18:43.023284", "status": "completed"} tags=[]
# ## Plot overview

# + papermill={"duration": 0.053447, "end_time": "2022-08-26T10:18:43.108989", "exception": false, "start_time": "2022-08-26T10:18:43.055542", "status": "completed"} tags=[]
nps = ds.time.size
nps

# + papermill={"duration": 0.667922, "end_time": "2022-08-26T10:18:43.795511", "exception": false, "start_time": "2022-08-26T10:18:43.127589", "status": "completed"} tags=[]
# prepare the animation
iz = 0
steps = [1, 5, 10, nps-1]

hca, hcb = arrange_axes(2,2, plot_cb=True, asp=1., fig_size_fac=1.5,
                        sharex=False, sharey=False, xlabel='x [km]', ylabel='y [km]')
ii=-1

for nn, ll in enumerate(steps):
    ii+=1; ax=hca[ii]; cax=hcb[ii]
    data = ds['ho'][ll,iz,:,:].compute()
    clim = 5e-2
    data[:ny//2,:] += -H0
    data[ny//2:,:] += -H0/2.
    hm = shade(ds.xt/1e3, ds.yt/1e3, data, ax=ax, cax=cax, clim=clim)
    ax.set_title('h [m]')
    ht = ax.set_title(f'{ds.time[ll].data/86400.:.1f}days', loc='right')

# + [markdown] papermill={"duration": 0.02007, "end_time": "2022-08-26T10:18:43.833496", "exception": false, "start_time": "2022-08-26T10:18:43.813426", "status": "completed"} tags=[]
# ## Make an animation

# + papermill={"duration": 0.070229, "end_time": "2022-08-26T10:18:43.922035", "exception": false, "start_time": "2022-08-26T10:18:43.851806", "status": "completed"} tags=[]
path_fig = f'{path_data}/'
fname_prf = run
fpath = f'{path_data}/shallowpy_combined.nc'

mfdset_kwargs = dict(combine='nested', concat_dim='time',
                     data_vars='minimal', coords='minimal', compat='override', join='override',
                    )
ds = xr.open_mfdataset(fpath, **mfdset_kwargs)
#ds = ds.compute()

# + papermill={"duration": 0.206921, "end_time": "2022-08-26T10:18:44.146548", "exception": false, "start_time": "2022-08-26T10:18:43.939627", "status": "completed"} tags=[]
# prepare the animation

iz = 0
ll=10

hca, hcb = arrange_axes(1,1, plot_cb=True, asp=1.00, fig_size_fac=3, axlab_kw=None,
                        sharex=False, sharey=False, xlabel='x [km]', ylabel='y [km]')
ii=-1
fig = plt.gcf()

ii+=1; ax=hca[ii]; cax=hcb[ii]
data = ds['ho'][ll,iz,:,:].compute()
data[:ny//2,:] += -H0
data[ny//2:,:] += -H0/2.
clim = 5e-2
hm = shade(ds.xt/1e3, ds.yt/1e3, data, ax=ax, cax=cax, clim=clim)
ax.set_title('h [m]')
ht = ax.set_title(f'{ds.time[ll].data/86400.:.1f}days', loc='right')


# + papermill={"duration": 0.054398, "end_time": "2022-08-26T10:18:44.220437", "exception": false, "start_time": "2022-08-26T10:18:44.166039", "status": "completed"} tags=[]
# function for updating the animation
def run(ll):
    print(f'll = {ll} / {ds.time.size}', end='\r')
    data = ds['ho'][ll,iz,:,:].data
    data[:ny//2,:] += -H0
    data[ny//2:,:] += -H0/2.
    hm[0].set_array(data.flatten())
    ht.set_text(f'{ds.time[ll].data/86400.:.1f}days')


# + papermill={"duration": 1.54999, "end_time": "2022-08-26T10:18:45.789239", "exception": false, "start_time": "2022-08-26T10:18:44.239249", "status": "completed"} tags=[]
# %%time
# --- save the animation
ani = animation.FuncAnimation(fig, run, ds.time.size)
if not os.path.exists(path_fig):
    os.mkdir(path_fig)
fpath_fig = f'{path_fig}/{fname_prf}.mp4'
print(f'Saving {fpath_fig}')
ani.save(fpath_fig, writer='ffmpeg', fps=40)

# + [markdown] papermill={"duration": 0.022032, "end_time": "2022-08-26T10:18:45.830832", "exception": false, "start_time": "2022-08-26T10:18:45.808800", "status": "completed"} tags=[]
# ## Showing the animation

# + papermill={"duration": 1.671754, "end_time": "2022-08-26T10:18:47.522191", "exception": false, "start_time": "2022-08-26T10:18:45.850437", "status": "completed"} tags=[]
# %%time
HTML(ani.to_jshtml())

# + papermill={"duration": 0.029695, "end_time": "2022-08-26T10:18:47.582490", "exception": false, "start_time": "2022-08-26T10:18:47.552795", "status": "completed"} tags=[]

# -


