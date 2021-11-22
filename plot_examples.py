# Combine all netcdf files
# ------------------------
mfdset_kwargs = dict(combine='nested', concat_dim='time',
                     data_vars='minimal', coords='minimal', compat='override', join='override',
                    )
ds = xr.open_mfdataset(f'{path_data}/*.nc', **mfdset_kwargs)
fpath = f'{path_data}/{file_prfx}_combined.nc'
print(f'Save file {fpath}')
ds.to_netcdf(fpath)

# --- here starts plotting
plt.close('all')

hop = ho[:,nspy:-nspy, nspx:-nspx]
uop = uo[:,nspy:-nspy, nspx:-nspx]
vop = vo[:,nspy:-nspy, nspx:-nspx]
Tuo = dict()
Tuo['tot'] = Tuo_tot[:,nspy:-nspy, nspx:-nspx]
Tuo['adv'] = Tuo_adv[:,nspy:-nspy, nspx:-nspx]
Tuo['dif'] = Tuo_dif[:,nspy:-nspy, nspx:-nspx]
Tuo['pgd'] = Tuo_pgr[:,nspy:-nspy, nspx:-nspx]
Tuo['cor'] = Tuo_cor[:,nspy:-nspy, nspx:-nspx]
Tvo = dict()
Tvo['tot'] = Tvo_tot[:,nspy:-nspy, nspx:-nspx]
Tvo['adv'] = Tvo_adv[:,nspy:-nspy, nspx:-nspx]
Tvo['dif'] = Tvo_dif[:,nspy:-nspy, nspx:-nspx]
Tvo['pgd'] = Tvo_pgr[:,nspy:-nspy, nspx:-nspx]
Tvo['cor'] = Tvo_cor[:,nspy:-nspy, nspx:-nspx]
Tho = dict()
Tho['tot'] = Tho_tot[:,nspy:-nspy, nspx:-nspx]
Tho['adv'] = Tho_adv[:,nspy:-nspy, nspx:-nspx]
Tho['dif'] = Tho_dif[:,nspy:-nspy, nspx:-nspx]

# --- time series
hca, hcb = pyic.arrange_axes(2,1, plot_cb=False, asp=0.5, fig_size_fac=1.5, sharex=False, sharey=False)
ii=-1

ii+=1; ax=hca[ii]; cax=hcb[ii]
ax.plot(times/86400., uo_ts, label='uo_ts')
ax.plot(times/86400., vo_ts, label='vo_ts')
ax.legend()

ii+=1; ax=hca[ii]; cax=hcb[ii]
ax.plot(times/86400., ho_ts, label='ho_ts')
ax.legend()

# --- mom. tend.
hca, hcb = pyic.arrange_axes(5,3, plot_cb=True, asp=1., fig_size_fac=1.)
ii=-1

clim = 'sym'

ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(xt/1e3, yt/1e3, uop[0,:,:], ax=ax, cax=cax, clim=clim)
#pyic.shade(xt/1e3, yt/1e3, vop[0,:,:], ax=ax, cax=cax, clim=clim)
ax.set_title(f'uo')

ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(xt/1e3, yt/1e3, hop[0,:,:]-H0, ax=ax, cax=cax, clim=clim)
ax.set_title(f'ho')

for nn, var in enumerate(['tot', 'adv', 'dif']):
  ii+=1; ax=hca[ii]; cax=hcb[ii]
  pyic.shade(xt/1e3, yt/1e3, Tho[var][0,:,:], ax=ax, cax=cax, clim=clim)
  ax.set_title(f'Tho_{var}')

for nn, var in enumerate(['tot', 'adv', 'dif', 'pgd', 'cor']):
  ii+=1; ax=hca[ii]; cax=hcb[ii]
  pyic.shade(xt/1e3, yt/1e3, Tuo[var][0,:,:], ax=ax, cax=cax, clim=clim)
  ax.set_title(f'Tuo_{var}')

for nn, var in enumerate(['tot', 'adv', 'dif', 'pgd', 'cor']):
  ii+=1; ax=hca[ii]; cax=hcb[ii]
  pyic.shade(xt/1e3, yt/1e3, Tvo[var][0,:,:], ax=ax, cax=cax, clim=clim)
  ax.set_title(f'Tvo_{var}')

#plt.show()
#sys.exit()

# --- fields
hca, hcb = pyic.arrange_axes(3,3, plot_cb=True, asp=1., fig_size_fac=1.5)
ii=-1

# --- ho
clim_ho = 1e-2

ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(xt/1e3, yt/1e3, ho0[0,:,:], ax=ax, cax=cax, clim=clim_ho)
ax.set_title('ho: initial conditions')

ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(xt/1e3, yt/1e3, hop[0,:,:], ax=ax, cax=cax, clim=clim_ho)
ax.set_title('ho: final conditions')

ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(xt/1e3, yt/1e3, (hop-ho0)[0,:,:], ax=ax, cax=cax, clim=clim_ho)
ax.set_title('ho: final - initial conditions')

# --- uo
clim_uo = 1.

ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(xt/1e3, yt/1e3, uo0[0,:,:], ax=ax, cax=cax, clim=clim_uo)
ax.set_title('uo: initial conditions')

ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(xt/1e3, yt/1e3, uop[0,:,:], ax=ax, cax=cax, clim=clim_uo)
ax.set_title('uo: final conditions')

ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(xt/1e3, yt/1e3, (uop-uo0)[0,:,:], ax=ax, cax=cax, clim=clim_uo)
ax.set_title('uo: final - initial conditions')

# --- vo
clim_vo = 1.

ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(xt/1e3, yt/1e3, vo0[0,:,:], ax=ax, cax=cax, clim=clim_vo)
ax.set_title('vo: initial conditions')

ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(xt/1e3, yt/1e3, vop[0,:,:], ax=ax, cax=cax, clim=clim_vo)
ax.set_title('vo: final conditions')

ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(xt/1e3, yt/1e3, (vop-vo0)[0,:,:], ax=ax, cax=cax, clim=clim_vo)
ax.set_title('vo: final - initial conditions')

# --- lines
hca, hcb = pyic.arrange_axes(2,3, plot_cb=False, asp=1., fig_size_fac=1.5, sharey=False, sharex=False)
ii=-1

clim_ho = 1.

# --- ho
ii+=1; ax=hca[ii]; cax=hcb[ii]
ax.plot(xt/1e3, ho0[0,ny//2,:], label='ho0')
ax.plot(xt/1e3, hop[0,ny//2,:], label='hop')
ax.set_title('ho [m]')
ax.set_xlabel('x [km]')

ii+=1; ax=hca[ii]; cax=hcb[ii]
ax.plot(yt/1e3, ho0[0,:,nx//2], label='ho0')
ax.plot(yt/1e3, hop[0,:,nx//2], label='hop')
ax.set_title('ho [m]')
ax.set_xlabel('y [km]')

# --- uo
ii+=1; ax=hca[ii]; cax=hcb[ii]
ax.plot(xt/1e3, uo0[0,ny//2,:], label='uo0')
ax.plot(xt/1e3, uop[0,ny//2,:], label='uop')
ax.set_title('uo [m]')
ax.set_xlabel('x [km]')

ii+=1; ax=hca[ii]; cax=hcb[ii]
ax.plot(yt/1e3, uo0[0,:,nx//2], label='uo0')
ax.plot(yt/1e3, uop[0,:,nx//2], label='uop')
ax.set_title('uo [m]')
ax.set_xlabel('y [km]')

# --- vo
ii+=1; ax=hca[ii]; cax=hcb[ii]
ax.plot(xt/1e3, vo0[0,ny//2,:], label='vo0')
ax.plot(xt/1e3, vop[0,ny//2,:], label='vop')
ax.set_title('vo [m]')
ax.set_xlabel('x [km]')

ii+=1; ax=hca[ii]; cax=hcb[ii]
ax.plot(yt/1e3, vo0[0,:,nx//2], label='vo0')
ax.plot(yt/1e3, vop[0,:,nx//2], label='vop')
ax.set_title('vo [m]')
ax.set_xlabel('y [km]')


plt.show()
