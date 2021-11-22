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

fpath = '/Users/nbruegge/work/movies/shallow_py/examp_kelvin_wave//test_????.nc'
ds = xr.open_mfdataset(fpath, **mfdset_kwargs)
ds = ds.compute()

H0 = 100.

var = 'ho'
iz = 0

plt.close('all')

# ---
hca, hcb = pyic.arrange_axes(1,1, plot_cb=True, asp=1., fig_size_fac=1.5, axlab_kw=None)
ii=-1

ll=10

#ii+=1; ax=hca[ii]; cax=hcb[ii]
#data = ds['uo'][ll,iz,:,:]
#hm = pyic.shade(ds.xt/1e3, ds.yt/1e3, data, ax=ax, cax=cax, clim='sym')
#ax.set_title('u [m/s]')
#
#ii+=1; ax=hca[ii]; cax=hcb[ii]
#data = ds['vo'][ll,iz,:,:]
#hm = pyic.shade(ds.xt/1e3, ds.yt/1e3, data, ax=ax, cax=cax, clim='sym')
#ax.set_title('v [m/s]')

ii+=1; ax=hca[ii]; cax=hcb[ii]
data = ds['ho'][ll,iz,:,:]
clim = [0,1e-2]
hm = pyic.shade(ds.xt/1e3, ds.yt/1e3, data-H0, ax=ax, cax=cax, clim=clim)
ax.set_title('h [m]')
ht = ax.set_title(f'{ds.time[ll].data/60.:.1f}min', loc='right')

#pyic.shade(ds.xt/1e3, ds.yt/1e3, (ds.ho[10,0,:,:]-H0).transpose(), ax=ax, cax=cax, clim='sym')

for ax in hca:
  ax.set_xlabel('x [km]')
  ax.set_ylabel('y [km]')

for ll in range(ds.time.size):
  data = ds[var][ll,iz,:,:].data - H0
  hm[0].set_array(data[1:,1:].flatten())
  ht.set_text(f'{ds.time[ll].data/60.:.1f}min')
  fpath_fig = f'/Users/nbruegge/work/movies/shallow_py/examp_kelvin_wave/anim_01/kelvin_wave_{ll:04d}.jpg'
  print(f'Saving figure {fpath_fig}')
  plt.savefig(fpath_fig, dpi=250)

plt.show()
