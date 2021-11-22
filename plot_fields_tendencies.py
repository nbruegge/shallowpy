
# --- mom. tend.
hca, hcb = pyic.arrange_axes(7,3, plot_cb=True, asp=1., fig_size_fac=1.,
                             sharex=True, sharey=True,
                            )
ii=-1

clim = 'sym'

pvars = ['tot', 'adv', 'dif', 'pgd', 'cor', 'vdf']

# --- uo
ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(xt/1e3, yt/1e3, uop[0,:,:], ax=ax, cax=cax, clim=clim)
ax.set_title(f'uo')

for nn, var in enumerate(pvars):
  ii+=1; ax=hca[ii]; cax=hcb[ii]
  pyic.shade(xt/1e3, yt/1e3, Tuo[var][0,:,:], ax=ax, cax=cax, clim=clim)
  ax.set_title(f'Tuo_{var}')

# --- vo
ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(xt/1e3, yt/1e3, vop[0,:,:], ax=ax, cax=cax, clim=clim)
ax.set_title(f'vo')

for nn, var in enumerate(pvars):
  ii+=1; ax=hca[ii]; cax=hcb[ii]
  pyic.shade(xt/1e3, yt/1e3, Tvo[var][0,:,:], ax=ax, cax=cax, clim=clim)
  ax.set_title(f'Tvo_{var}')

# --- ho
ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(xt/1e3, yt/1e3, hop[0,:,:]-H0, ax=ax, cax=cax, clim=clim)
ax.set_title(f'ho')

#pvars = ['tot', 'adv', 'dif']
pvars = ['tot', 'adv', 'dif', 'mix', 'con']
for nn, var in enumerate(pvars):
  ii+=1; ax=hca[ii]; cax=hcb[ii]
  pyic.shade(xt/1e3, yt/1e3, Tho[var][0,:,:], ax=ax, cax=cax, clim=clim)
  ax.set_title(f'Tho_{var}')

