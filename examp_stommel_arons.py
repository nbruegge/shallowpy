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

fac = 1.
nx = int(250*fac)
ny = int(1000*fac)
#nt = int(1600*fac)
#nt = 960*30
#nt = 8640
nt = 1000000
#nt = 100

file_prfx = 'test2'

picture_frequency = 0
output_frequency = 500
diagnostic_frequency = output_frequency
diagnostic_frequency = 50

dx = 10e3/fac
dy = dx
y0 = ny*dy/2.

grav = 0.02
rho = np.array([1000.])
#rho = np.array([1., 2., 3.])
nz = rho.size

H0 = 400.
cph = np.sqrt(grav*H0)
dist = dt*nt * cph
#dt  = 0.1*dx/np.sqrt(grav*H0)
#dt = 225./fac # for fac = 0.25
dt = 600.
#dt = 1800.

nspx = 1
nspy = 1
epsab = 0.01

kh = 1e4*fac**2
Ah = kh
drag_coeff_linear = 2e-6
lam_ho = 1e-9

#lat_0 = 30.
#omega = 2*np.pi/(24.*3600.)     # Earth's angular frequency [s**-1]
#R = 6.371e6                     # Earth's radius [m]
#f0 = 2*omega*np.sin(lat_0*np.pi/180.)
#beta = 2*omega/R*np.cos(lat_0*np.pi/180.)
f0 = 0.
beta = 2.3e-11

print(f'Munk layer:    {(Ah/beta)**(1./3.)/1e3}km') 
print(f'Stommel layer: {drag_coeff_linear/beta/1e3}km')
if dx>2.*drag_coeff_linear/beta:
  print(f'Resolution too small to resolve Stommel layer.')
  sys.exit()

do_momentum_advection = False                   
do_momentum_diffusion = True
do_momentum_drag = True
do_momentum_coriolis_exp = False
do_momentum_coriolis_imp = True
do_momentum_pressure_gradient = True
do_momentum_windstress = False
do_height_diffusion = False
do_height_advection = True
do_height_stommel_arons = True

# Initialize the grid and initial conditions
# ------------------------------------------
exec(open('./shallowpy_grid_setup.py').read())

# Modify initial conditions
# -------------------------
#ho0 = 0.01*(Xt-Lx/2.)/Lx
#ho0 = H0+0.1*np.sin(Xt/(Lx+dx)*2*np.pi*2)
#ho0 = H0+0.1*np.exp(-((Xt-0.5*Lx)**2+(Yt-0.5*Ly)**2)/(1.e-3*(Lx**2+Ly**2)))
#ho0 = ho0[np.newaxis,:,:]
#H0 = 0.
#ho0 += H0

#uo0 = cph + 0.*Xu
#uo0 = uo0[np.newaxis,:,:]

maskt0[:,:,0] = 0.
maskt0[:,:,-1] = 0.
maskt0[:,0,:] = 0.
maskt0[:,-1,:] = 0.

eta0[0,:,:] = 0.
#eta0[1,:,:] = -10.
#eta0[2,:,:] = -40.
#eta0[3,:,:] = -100.
eta0[nz,:,:] = -H0
ho0 = eta0[:-1,:,:]-eta0[1:,:,:]

#ho0 *= maskt[:,nspy:-nspy,nspx:-nspy]
ho0 *= maskt0

#taux0 = -1e-4*np.cos(2.*np.pi*Yt/Ly * 0.5)
#taux0 = -1e-4*np.cos(2.*np.pi*Yt/Ly)

ny_dw = 32
dw_source0 = np.zeros((nz,ny,nx))
dw_source0[:,-5:,:] = 5e6/((nx-2)*dx*ny_dw*dy)
dw_source0 *= maskt0

ix = np.array([nx//2])
iy = np.array([ny//2])

# Run the model
# -------------
exec(open('./shallowpy_main.py').read())

# Do post-processing
# ------------------
exec(open('./pp_main.py').read())

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
  ds['ho'] = ds.ho.where(masktp==1)
  ds['uo'] = ds.ho.where(maskup==1)
  ds['vo'] = ds.ho.where(maskvp==1)
  ds = ds.compute()
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

# ---
hca, hcb = pyic.arrange_axes(1,1, plot_cb=True, asp=2., fig_size_fac=3.)
ii=-1

ii+=1; ax=hca[ii]; cax=hcb[ii]
hm = pyic.shade(xt/1e3, yt/1e3, hop[0,:,:]-hop[0,:,:].mean(), ax=ax,cax=cax, conts='auto', clim='sym')
ax.set_title('layer thickness ho [m]')
ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')

# ---
hca, hcb = pyic.arrange_axes(1,2, plot_cb=False, asp=0.5, fig_size_fac=1.5)
ii=-1

ii+=1; ax=hca[ii]; cax=hcb[ii]
ax.plot(yt/1e3, hop[0,:,:].mean(dim='xt'))
ax.set_title('zonally averaged ho [m]')

ii+=1; ax=hca[ii]; cax=hcb[ii]
transp = vop[0,:,:].data*hop[0,:,:]*dy
xbc = 500e3
ax.plot(yt/1e3, transp.sum(dim='xt')/1e6, label='total transport')
ax.plot(yt/1e3, transp.where(Xt<=xbc).sum(dim='xt')/1e6, label='BC transport')
ax.plot(yt/1e3, transp.where(Xt>xbc).sum(dim='xt')/1e6, label='interior transport')
ax.set_title('transport [Sv]')
ax.set_xlabel('y [km]')
ax.legend()

plt.show()
