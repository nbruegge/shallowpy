if mode=='default_parameters':
  nx = 10
  ny = 10
elif mode=='initial_conditions':
  ho0 = 0.01*np.exp(-((Xt-0.5*Lx)**2+(Yt-0.5*Ly)**2)/(1.e-3*(Lx**2+Ly**2)))
  
  ho0 = ho0[np.newaxis,:,:]
  H0 = 0.
  ho0 += H0
else:
  raise ValueError(f'::: Error: Unknown mode {mode}.:::')
