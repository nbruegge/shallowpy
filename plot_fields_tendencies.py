hop = ho[:,nspy:-nspy, nspx:-nspx]
uop = uo[:,nspy:-nspy, nspx:-nspx]
vop = vo[:,nspy:-nspy, nspx:-nspx]
Tuo = dict()
Tuo['tot'] = Tuo_tot[:,nspy:-nspy, nspx:-nspx]
Tuo['adv'] = Tuo_adv[:,nspy:-nspy, nspx:-nspx]
Tuo['dif'] = Tuo_dif[:,nspy:-nspy, nspx:-nspx]
Tuo['pgd'] = Tuo_pgr[:,nspy:-nspy, nspx:-nspx]
Tuo['cor'] = Tuo_cor[:,nspy:-nspy, nspx:-nspx]
Tuo['vdf'] = Tuo_vdf[:,nspy:-nspy, nspx:-nspx]
Tvo = dict()
Tvo['tot'] = Tvo_tot[:,nspy:-nspy, nspx:-nspx]
Tvo['adv'] = Tvo_adv[:,nspy:-nspy, nspx:-nspx]
Tvo['dif'] = Tvo_dif[:,nspy:-nspy, nspx:-nspx]
Tvo['pgd'] = Tvo_pgr[:,nspy:-nspy, nspx:-nspx]
Tvo['cor'] = Tvo_cor[:,nspy:-nspy, nspx:-nspx]
Tvo['vdf'] = Tvo_vdf[:,nspy:-nspy, nspx:-nspx]
Tho = dict()
Tho['tot'] = Tho_tot[:,nspy:-nspy, nspx:-nspx]
Tho['adv'] = Tho_adv[:,nspy:-nspy, nspx:-nspx]
Tho['dif'] = Tho_dif[:,nspy:-nspy, nspx:-nspx]
Tho['mix'] = Tho_mix[:,nspy:-nspy, nspx:-nspx]
Tho['con'] = Tho_con[:,nspy:-nspy, nspx:-nspx]
Tho['for'] = Tho_for[:,nspy:-nspy, nspx:-nspx]

# --- mom. tend.
hca, hcb = arrange_axes(7,3, plot_cb=True, asp=1., fig_size_fac=1.,
                             sharex=True, sharey=True,
                            )
ii=-1

clim = 'sym'

pvars = ['tot', 'adv', 'dif', 'pgd', 'cor', 'vdf']

# --- uo
ii+=1; ax=hca[ii]; cax=hcb[ii]
shade(xt/1e3, yt/1e3, uop[0,:,:], ax=ax, cax=cax, clim=clim)
ax.set_title(f'uo')

for nn, var in enumerate(pvars):
  ii+=1; ax=hca[ii]; cax=hcb[ii]
  shade(xt/1e3, yt/1e3, Tuo[var][0,:,:], ax=ax, cax=cax, clim=clim)
  ax.set_title(f'Tuo_{var}')

# --- vo
ii+=1; ax=hca[ii]; cax=hcb[ii]
shade(xt/1e3, yt/1e3, vop[0,:,:], ax=ax, cax=cax, clim=clim)
ax.set_title(f'vo')

for nn, var in enumerate(pvars):
  ii+=1; ax=hca[ii]; cax=hcb[ii]
  shade(xt/1e3, yt/1e3, Tvo[var][0,:,:], ax=ax, cax=cax, clim=clim)
  ax.set_title(f'Tvo_{var}')

# --- ho
ii+=1; ax=hca[ii]; cax=hcb[ii]
shade(xt/1e3, yt/1e3, hop[0,:,:]-H0, ax=ax, cax=cax, clim=clim)
ax.set_title(f'ho')

#pvars = ['tot', 'adv', 'dif']
pvars = ['tot', 'adv', 'dif', 'mix', 'con']
for nn, var in enumerate(pvars):
  ii+=1; ax=hca[ii]; cax=hcb[ii]
  shade(xt/1e3, yt/1e3, Tho[var][0,:,:], ax=ax, cax=cax, clim=clim)
  ax.set_title(f'Tho_{var}')

