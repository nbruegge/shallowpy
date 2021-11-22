# Grid setup
# ----------
xu = np.arange(-x0, nx*dx-x0, dx)
yu = np.arange(-y0, ny*dy-y0, dy)
xt = np.concatenate((0.5*(xu[1:]+xu[:-1]),[xu[-1]+dx*0.5]))
yt = np.concatenate((0.5*(yu[1:]+yu[:-1]),[yu[-1]+dy*0.5]))

Xt, Yt = np.meshgrid(xt, yt)
Xu, Yu = np.meshgrid(xu, yu)

Lx = (xu[-1]-xu[0])
Ly = (yu[-1]-yu[0])

ft0 = f0+beta*(Yt)
fu0 = f0+beta*(Yt)
fv0 = f0+beta*(Yu)
ft0 = ft0[np.newaxis,:,:]
fu0 = fu0[np.newaxis,:,:]
fv0 = fv0[np.newaxis,:,:]

gprime = grav*np.concatenate(([1], (rho[1:]-rho[:-1])/rho[0]))
gprime = gprime[:,np.newaxis,np.newaxis]

# Initial conditions
# ------------------
uo0 = np.ma.zeros((nz,ny,nx))
vo0 = np.ma.zeros((nz,ny,nx))
ho0 = np.ma.zeros((nz,ny,nx))
eta0 = np.ma.zeros((nz+1,ny,nx))

# Ocean mask
# ----------
maskt0 = np.ones((nz,ny,nx))

# Wind stress
# -----------
taux0 = np.ma.zeros((ny,nx))
tauy0 = np.ma.zeros((ny,nx))

# Forcing
# -------
def ho_forcing(time):
  Tho_for = 0.
  return Tho_for
def uo_forcing(time):
  Tuo_for = 0.
  return Tuo_for
def vo_forcing(time):
  Tvo_for = 0.
  return Tvo_for

# Time series
# -----------
times = np.zeros((nt))
uo_ts = np.zeros((nt, ix.size))
vo_ts = np.zeros((nt, ix.size))
ho_ts = np.zeros((nt, ix.size))
