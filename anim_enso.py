import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import pyicon as pyic
import argparse
import matplotlib.animation as animation

path_data = '/Users/nbruegge/work/movies/shallow_py/examp_enso/'
path_fig = f'{path_data}/anim_01/'
fname_prf = 'kelvin_wave'
fpath = f'{path_data}/test_combined.nc'

mfdset_kwargs = dict(combine='nested', concat_dim='time', 
                     data_vars='minimal', coords='minimal', compat='override', join='override',
                    )
ds = xr.open_mfdataset(fpath, **mfdset_kwargs)
#ds = ds.compute()


H0 = 500.
#H0 = 0.
nt = ds.time.size
var = 'ho'
iz = 0
ll=110

plt.close('all')

# ---
hca, hcb = pyic.arrange_axes(1,1, plot_cb=True, asp=1., fig_size_fac=2.0, axlab_kw=None,
  dfigl=0.2, dfigr=0.2)
fig = plt.gcf()
ii=-1

ii+=1; ax=hca[ii]; cax=hcb[ii]
data = ds['ho'][ll,iz,:,:]
clim = 3e-2
hm = pyic.shade(ds.xt/1e3, (ds.yt-ds.yt[-1]/2.)/1e3, data-H0, ax=ax, cax=cax, clim=clim)
ax.set_title('h [m]')
ht = ax.set_title(f'{ds.time[ll].data/86400:.1f}days', loc='right')

for ax in hca:
  ax.set_xlabel('x [km]')
  ax.set_ylabel('y [km]')

#plt.show()
#sys.exit()

def run(ll):
  print(f'll = {ll} / {ds.time.size}', end='\r')
  data = ds[var][ll,iz,:,:].data - H0
  #hm[0].set_array(data[1:,1:].flatten())
  hm[0].set_array(data.flatten())
  ht.set_text(f'{ds.time[ll].data/86400.:.1f}days')
  #fpath_fig = f'{path_fig}/{fname_prf}_{ll:04d}.jpg'
  #print(f'Saving figure {fpath_fig}')
  #plt.savefig(fpath_fig, dpi=250)
  

#for ll in range(ds.time.size):
#  data = ds[var][ll,iz,:,:].data - H0
#  #hm[0].set_array(data[1:,1:].flatten())
#  hm[0].set_array(data.flatten())
#  ht.set_text(f'{ds.time[ll].data/86400.:.1f}min')
#  fpath_fig = f'{path_fig}/{fname_prf}_{ll:04d}.jpg'
#  print(f'Saving figure {fpath_fig}')
#  plt.savefig(fpath_fig, dpi=250)

ani = animation.FuncAnimation(fig, run, ds.time.size)
if not os.path.exists(path_fig):
  os.mkdir(path_fig)
ani.save('test.mp4', writer='ffmpeg', fps=40)
#plt.show()
