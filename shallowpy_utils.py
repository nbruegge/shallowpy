
def bnd_exch(phi, nspx=1, nspy=1):
  phi[:,0:nspy,:] = phi[:,-2*nspy:-nspy:,:]
  phi[:,-nspy:,:] = phi[:,nspy:2*nspy,:]
  phi[:,:,0:nspx] = phi[:,:,-2*nspx:-nspx:]
  phi[:,:,-nspx:] = phi[:,:,nspx:2*nspx]
  return phi

def create_output_var(nparr, dims, time=0):
  xrarr = xr.DataArray(nparr[:, nspy:-nspy, nspx:-nspx], dims=dims)
  #xrarr = xr.DataArray(nparr[np.newaxis, :, nspy:-nspy, nspx:-nspx], dims=dims)
  #xrarr['time'] = time
  return xrarr

def timing(time_wc_ini, verbose=False):
  time_wc_now = datetime.datetime.now()
  #dt_wc = datetime.timedelta(seconds=(time_wc_now - time_wc_ini).total_seconds()/nnp)
  dt_wc = (time_wc_now - time_wc_ini)/(nnp-1) * nt/output_frequency
  time_wc_fin = time_wc_ini + dt_wc
  if verbose:
    print(f'nnp = {nnp}, Total run time: {dt_wc.total_seconds()/60:.1f}min, done at {time_wc_fin}')
  return dt_wc, time_wc_fin
