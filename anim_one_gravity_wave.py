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

fpath = '/Users/nbruegge/work/movies/shallow_py/examp_short_gravity_waves//test_????.nc'
ds = xr.open_mfdataset(fpath, **mfdset_kwargs)
ds = ds.compute()

H0 = 1000.

var = 'uo'
iz = 0

plt.close('all')

# ---
hca, hcb = pyic.arrange_axes(3,1, plot_cb=True, asp=1., fig_size_fac=1.5, axlab_kw=None)
ii=-1

ll=10

ii+=1; ax=hca[ii]; cax=hcb[ii]
data = ds['uo'][ll,iz,:,:]
hm = pyic.shade(ds.xt/1e3, ds.yt/1e3, data, ax=ax, cax=cax, clim='sym')
ax.set_title('u [m/s]')
ht = ax.set_title(f'{ds.time[ll].data/60.:.1f}min', loc='right')

ii+=1; ax=hca[ii]; cax=hcb[ii]
data = ds['vo'][ll,iz,:,:]
hm = pyic.shade(ds.xt/1e3, ds.yt/1e3, data, ax=ax, cax=cax, clim='sym')
ax.set_title('v [m/s]')

ii+=1; ax=hca[ii]; cax=hcb[ii]
data = ds['ho'][ll,iz,:,:]
hm = pyic.shade(ds.xt/1e3, ds.yt/1e3, data, ax=ax, cax=cax, clim='auto')
ax.set_title('h [m]')

#pyic.shade(ds.xt/1e3, ds.yt/1e3, (ds.ho[10,0,:,:]-H0).transpose(), ax=ax, cax=cax, clim='sym')

for ax in hca:
  ax.set_xlabel('x [km]')
  ax.set_ylabel('y [km]')

for ll in range(ds.time.size):
  data = ds[var][ll,iz,:,:].data
  hm[0].set_array(data[1:,1:].flatten())
  ht.set_text(f'{ds.time[ll].data/60.:.1f}min')
  fpath_fig = f'/Users/nbruegge/work/movies/shallow_py/examp_long_gravity_wave/anim_01/two_waves_{ll:04d}.jpg'
  print(f'Saving figure {fpath_fig}')
  plt.savefig(fpath_fig, dpi=250)

plt.show()
