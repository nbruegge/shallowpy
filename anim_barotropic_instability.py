import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import pyicon as pyic
import argparse

mfdset_kwargs = dict(combine='nested', concat_dim='time', 
                     data_vars='minimal', coords='minimal', compat='override', join='override',
                    )

run = 'examp_barotropic_instability_03'
fpath = f'/work/mh0033/m300602/shallowpy/{run}/test_?[0-5]??.nc'
ds = xr.open_mfdataset(fpath, **mfdset_kwargs)
ds = ds.compute()

H0 = 100.

iz = 0

plt.close('all')

# ---
hca, hcb = pyic.arrange_axes(2,2, plot_cb=True, asp=1., fig_size_fac=1.5, axlab_kw=None, sharex=True, sharey=True, xlabel='x [km]', ylabel='y [km]')
ii=-1

ll=70

ii+=1; ax=hca[ii]; cax=hcb[ii]
data = ds['uo'][ll,iz,:,:]
hm1 = pyic.shade(ds.xt/1e3, ds.yt/1e3, data, ax=ax, cax=cax, clim=20)
ax.set_title('u [m/s]')

ii+=1; ax=hca[ii]; cax=hcb[ii]
data = ds['vo'][ll,iz,:,:]
hm2 = pyic.shade(ds.xt/1e3, ds.yt/1e3, data, ax=ax, cax=cax, clim=20)
ax.set_title('v [m/s]')

ii+=1; ax=hca[ii]; cax=hcb[ii]
data = ds['ho'][ll,iz,:,:]
hm3 = pyic.shade(ds.xt/1e3, ds.yt/1e3, data-H0, ax=ax, cax=cax, clim=20)
ax.set_title('h [m]')
ht = ax.set_title(f'{ds.time[ll].data/3600.:.1f}h', loc='right')

ny, nx = data.shape
dx = ds.xt[1].data-ds.xt[0].data
dy = ds.yt[1].data-ds.yt[0].data
vort = np.ma.zeros((ny,nx))
vo = ds.vo[ll,iz,:,:].data
uo = ds.uo[ll,iz,:,:].data
vort[1:,1:] = (vo[:-1,1:]-vo[:-1,:-1])/dx - (uo[1:,:-1]-uo[:-1,:-1])/dy

ii+=1; ax=hca[ii]; cax=hcb[ii]
hm4 = pyic.shade(ds.xt/1e3, ds.yt/1e3, vort, ax=ax, cax=cax, clim=5e-4)
ax.set_title('vorticity [1/s]')

#plt.show()
#sys.exit()

for ll in range(ds.time.size):
  uo = ds['uo'][ll,iz,:,:].data
  vo = ds['vo'][ll,iz,:,:].data
  ho = ds['ho'][ll,iz,:,:].data - H0
  vort = np.ma.zeros((ny,nx))
  vo = ds.vo[ll,iz,:,:].data
  uo = ds.uo[ll,iz,:,:].data
  vort[1:,1:] = (vo[:-1,1:]-vo[:-1,:-1])/dx - (uo[1:,:-1]-uo[:-1,:-1])/dy
  hm1[0].set_array(uo.flatten())
  hm2[0].set_array(vo.flatten())
  hm3[0].set_array(ho.flatten())
  hm4[0].set_array(vort.flatten())
  ht.set_text(f'{ds.time[ll].data/60.:.1f}min')
  fpath_fig = f'/work/mh0033/m300602/shallowpy/movies/examp_barotropic_instability/anim_02/barotropic_instability_{ll:04d}.jpg'
  print(f'Saving figure {fpath_fig}')
  plt.savefig(fpath_fig, dpi=250)

plt.show()
