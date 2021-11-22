# --- time series
hca, hcb = pyic.arrange_axes(2,1, plot_cb=False, asp=0.5, fig_size_fac=1.5, sharex=False, sharey=False)
ii=-1

ii+=1; ax=hca[ii]; cax=hcb[ii]
ax.plot(times/86400., uo_ts, label='uo_ts')
ax.plot(times/86400., vo_ts, label='vo_ts')
ax.legend()
ax.set_title('velocity [m/s]')

ii+=1; ax=hca[ii]; cax=hcb[ii]
ax.plot(times/86400., ho_ts, label='ho_ts')
ax.legend()
ax.set_title('layer thickness [m]')

for ax in hca:
  ax.set_xlabel('time [days]')
