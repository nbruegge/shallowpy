uo0 = xr.DataArray(uo0, dims=dimsu)
vo0 = xr.DataArray(vo0, dims=dimsv)
ho0 = xr.DataArray(ho0, dims=dimst)
maskup = masku[:,nspy:-nspy, nspx:-nspx]
maskvp = maskv[:,nspy:-nspy, nspx:-nspx]
masktp = maskt[:,nspy:-nspy, nspx:-nspx]
uo0 = uo0.where(maskup==1)
vo0 = vo0.where(maskvp==1)
ho0 = ho0.where(masktp==1)

# --- fields
hca, hcb = arrange_axes(3,3, plot_cb=True, asp=1., fig_size_fac=1.5)
ii=-1

# --- ho
#clim_ho = 1e-2
clim_ho = 'sym'

ii+=1; ax=hca[ii]; cax=hcb[ii]
shade(xt/1e3, yt/1e3, ho0[0,:,:]-H0, ax=ax, cax=cax, clim=clim_ho)
ax.set_title('ho: initial conditions')

ii+=1; ax=hca[ii]; cax=hcb[ii]
shade(xt/1e3, yt/1e3, hop[0,:,:]-H0, ax=ax, cax=cax, clim=clim_ho)
ax.set_title('ho: final conditions')

ii+=1; ax=hca[ii]; cax=hcb[ii]
shade(xt/1e3, yt/1e3, (hop-ho0)[0,:,:], ax=ax, cax=cax, clim=clim_ho)
ax.set_title('ho: final - initial conditions')

# --- uo
#clim_uo = 1.
clim_uo = 'sym'

ii+=1; ax=hca[ii]; cax=hcb[ii]
shade(xt/1e3, yt/1e3, uo0[0,:,:], ax=ax, cax=cax, clim=clim_uo)
ax.set_title('uo: initial conditions')

ii+=1; ax=hca[ii]; cax=hcb[ii]
shade(xt/1e3, yt/1e3, uop[0,:,:], ax=ax, cax=cax, clim=clim_uo)
ax.set_title('uo: final conditions')

ii+=1; ax=hca[ii]; cax=hcb[ii]
shade(xt/1e3, yt/1e3, (uop-uo0)[0,:,:], ax=ax, cax=cax, clim=clim_uo)
ax.set_title('uo: final - initial conditions')

# --- vo
#clim_vo = 1.
clim_vo = 'sym'

ii+=1; ax=hca[ii]; cax=hcb[ii]
shade(xt/1e3, yt/1e3, vo0[0,:,:], ax=ax, cax=cax, clim=clim_vo)
ax.set_title('vo: initial conditions')

ii+=1; ax=hca[ii]; cax=hcb[ii]
shade(xt/1e3, yt/1e3, vop[0,:,:], ax=ax, cax=cax, clim=clim_vo)
ax.set_title('vo: final conditions')

ii+=1; ax=hca[ii]; cax=hcb[ii]
shade(xt/1e3, yt/1e3, (vop-vo0)[0,:,:], ax=ax, cax=cax, clim=clim_vo)
ax.set_title('vo: final - initial conditions')
