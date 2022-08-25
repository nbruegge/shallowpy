# Namelist
# --------
nx = 120
ny = 120
nt = 1000
#nt = 20
x0 = 0.
y0 = 0.

picture_frequency = 0
output_frequency = 10
diagnostic_frequency = 100

dx = 10e3
dy = dx
dt = 360.
#dt = 360.

grav = 9.81
#rho = np.array([1024., 1028.])
rho = np.array([1024.])
nz = rho.size

#maskt0 = np.ones((nz,ny,nx))
#maskt0[:,:,0] = 0.
#maskt0[:,:,-1] = 0.
##maskt0[:,0,:] = 0.
##maskt0[:,-1,:] = 0.

nspx = nspy = 1
epsab = 0.01

kh = 20.
Ah = kh

f0 = 1e-4
beta = 0.*1e-11
Y0 = 0.

ix = np.array([0])
iy = np.array([0])

#U0 = 0.5
#cfl = dt/dx*U0
#print(f'cfl = {cfl}')

# Output
# ------
run = 'test'
path_data = f'/Users/nbruegge/work/movies/shallow_py/{run}/'

fig_prfx = run 
nnf=0
file_prfx = run
nnp=0

do_height_advection = True
do_upwind_advection = True
do_linear_height_advection = False
do_height_diffusion = True
do_height_stommel_arons = False
do_height_forcing = False
do_momentum_drag = False
do_momentum_advection = True
do_momentum_diffusion = True
do_momentum_coriolis_imp = True
do_momentum_coriolis_exp = False
do_momentum_pressure_gradient = True
do_momentum_windstress = True
do_momentum_forcing = False
