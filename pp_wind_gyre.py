import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import pyicon as pyic
import argparse
from shallowpy_utils import create_output_var

file_prfx = 'test'
#fpath = f'{path_data}/{file_prfx}_combined.nc'
path_data = f'/Users/nbruegge/work/movies/shallow_py/examp_wind_gyre/'
fpath = '/Users/nbruegge/work/movies/shallow_py/examp_wind_gyre//test_combined.nc'
print(f'Load file {fpath}')
#ds = xr.open_dataset(fpath)

mfdset_kwargs = dict(combine='nested', concat_dim='time',
                     data_vars='minimal', coords='minimal', compat='override', join='override',
                    )
ds = xr.open_mfdataset(f'{path_data}/{file_prfx}_00[0-5]?.nc', **mfdset_kwargs)

xt = ds.xt
yt = ds.yt
H0 = 500.

ds['ho'] = ds.ho.where(ds.ho!=0)

#exec(open('./pp_main.py').read())

plt.close('all')

# ---
hca, hcb = pyic.arrange_axes(3,2, plot_cb=True, asp=1., fig_size_fac=1.5)
ii=-1

it = np.array([4, 8, 12, 16, 20])

for ll in range(it.size):
  clim = 'sym'

  ii+=1; ax=hca[ii]; cax=hcb[ii]
  data = ds.ho[it[ll],0,:,:]-H0
  pyic.shade(xt/1e3, yt/1e3, data, ax=ax, cax=cax, clim=clim)
  ax.set_title(f'time = {data.time.data/86400.:.1f}days')

# ---
hca, hcb = pyic.arrange_axes(1,1, plot_cb=False, asp=1., fig_size_fac=1.5, xlabel='time [days]')
ii=-1

ii+=1; ax=hca[ii]; cax=hcb[ii]
data = ds.ho[:,0,:,:].sel(xt=70e3, yt=4200e3, method='nearest')
ax.plot(data.time/86400, data)


plt.show()
