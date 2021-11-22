import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import pyicon as pyic
import datetime

#class Settings(object):
#  def __init__(self):
#    return
#
#D = Settings()
#D.nx = 128
#D.ny = 128
#
#Set = Settings()
#Set.nx = 10
#
#
#for var in vars(Set_def).keys():
#  if var in vars(Set_user).keys():
#    setattr(D, var, getattr(Set, var))
#
##Set = set_parameter(Set, 'nx', 124)
##
##def set_parameter(Set, name, val):
##  Set[name] 
#
#
#sys.exit()
## -----------------

def bnd_exch(phi):
  phi[:,0:nspy,:] = phi[:,-2*nspy:-nspy:,:]
  phi[:,-nspy:,:] = phi[:,nspy:2*nspy,:]
  phi[:,:,0:nspx] = phi[:,:,-2*nspx:-nspx:]
  phi[:,:,-nspx:] = phi[:,:,nspx:2*nspx]
  return phi

def create_output_var(nparr, dims, time=0):
  xrarr = xr.DataArray(nparr[:, nspy:-nspy, nspx:-nspx], dims=dims)
  xrarr = xrarr.where(xrarr!=0.)
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

## Namelist
## --------
#nx = 120
#ny = 120
#nt = 1000
##nt = 20
#
#picture_frequency = 0
#output_frequency = 10
#
#dx = 10e3
#dy = dx
#dt = 360.
##dt = 360.
#
#grav = 9.81
##rho = np.array([1024., 1028.])
#rho = np.array([1024.])
#nz = rho.size
#
#maskt0 = np.ones((nz,ny,nx))
#maskt0[:,:,0] = 0.
#maskt0[:,:,-1] = 0.
##maskt0[:,0,:] = 0.
##maskt0[:,-1,:] = 0.
#
#nspx = nspy = 1
#epsab = 0.01
#
#kh = 20.
#Ah = kh
#
#f0 = 1e-4
#beta = 0.*1e-11
#
##U0 = 0.5
##cfl = dt/dx*U0
##print(f'cfl = {cfl}')
#
## Output
## ------
#run = 'test2'
#path_data = f'/Users/nbruegge/work/movies/shallow_py/{run}/'
#
#fig_prfx = run 
#nnf=0
#file_prfx = run
#nnp=0
#
#do_height_advection = True
#do_height_diffusion = True
#do_momentum_advection = False
#do_momentum_diffusion = False
#do_momentum_coriolis = True
#do_momentum_pressure_gradient = True

## Grid setup
## ----------
#xu = np.arange(0., nx*dx, dx)
#yu = np.arange(0., ny*dy, dy)
#xt = np.concatenate((0.5*(xu[1:]+xu[:-1]),[xu[-1]+dx*0.5]))
#yt = np.concatenate((0.5*(yu[1:]+yu[:-1]),[yu[-1]+dy*0.5]))
#
#Xt, Yt = np.meshgrid(xt, yt)
#Xu, Yu = np.meshgrid(xu, yu)
#
#Lx = xu[-1]
#Ly = yu[-1]
#
#ft0 = f0+beta*Yt
#fu0 = f0+beta*Yu
#ft0 = ft0[np.newaxis,:,:]
#fu0 = fu0[np.newaxis,:,:]
#
#gprime = grav*np.concatenate(([1], (rho[1:]-rho[:-1])/rho[0]))
#gprime = gprime[:,np.newaxis,np.newaxis]

## time series
## -----------
#times = np.zeros((nt))
#uo_ts = np.zeros((nt))
#vo_ts = np.zeros((nt))
#ho_ts = np.zeros((nt))
#
#ix = np.argmin((xt-328e3)**2)
#iy = np.argmin((yt-295e3)**2)

## Initial conditions
## ------------------
#uo0 = np.ma.zeros((nz,ny,nx))
#vo0 = np.ma.zeros((nz,ny,nx))
#ho0 = np.ma.zeros((nz,ny,nx))
#eta_bot0 = np.ma.zeros((nz,ny,nx))

#uo0 += U0
#vo0 += U0
# --- large shifted blob
#ho0 = 0.01*np.exp(-((Xt-0.5*Lx)**2+(Yt-0.5*Ly)**2)/(1.e-2*(Lx**2+Ly**2)))
#hotmp = np.concatenate((ho0[:,nx*3//4:],ho0[:,:nx*3//4]),axis=1)
#ho0 = hotmp
# --- narrow blob
#ho0 = 0.01*np.exp(-((Xt-0.5*Lx)**2+(Yt-0.5*Ly)**2)/(1.e-3*(Lx**2+Ly**2)))
# --- single front
#ho0 = (Xt-Lx/2.)/Lx
#H0 = 0.2
# --- jet
#ho0 = 0.01*np.exp(-((Xt-0.5*Lx)**2)/(1.e-3*(Lx**2+Ly**2)))

#ho0 = ho0[np.newaxis,:,:]
#H0 = 0.
#ho0 += H0
#uo0 = np.exp(-((Xu-0.5*Lx)**2+(Yt-0.5*Ly)**2)/(1.e-3*(Lx**2+Ly**2)))
#uo0 = uo0[np.newaxis,:,:]
#vo0 = np.exp(-((Xt-0.5*Lx)**2+(Yu-0.5*Ly)**2)/(1.e-3*(Lx**2+Ly**2)))
#vo0 = vo0[np.newaxis,:,:]
#psi0 = np.exp(-((Xu-0.5*Lx)**2+(Yu-0.5*Ly)**2)/(1.e-3*(Lx**2+Ly**2)))
#uo0 = np.zeros((ny,nx))
#vo0 = np.zeros((ny,nx))
#uo0[:-1,:-1] =   (psi0[1:,:-1]-psi0[:-1,:-1])/dy
#vo0[:-1,:-1] = - (psi0[:-1,1:]-psi0[:-1,:-1])/dx
#uo0 = uo0[np.newaxis,:,:]*1e4
#vo0 = vo0[np.newaxis,:,:]*1e4

if not os.path.isdir(path_data):
  print(f'Creating directory {path_data}.')
  os.mkdir(path_data)

# Initialization
# --------------

maskt = np.ones((nz,ny+2*nspy,nx+2*nspx))
masku = np.ones((nz,ny+2*nspy,nx+2*nspx))
maskv = np.ones((nz,ny+2*nspy,nx+2*nspx))
maskt[:,nspy:-nspy,nspx:-nspx] = maskt0
maskt = bnd_exch(maskt)
masku[:,:,1:] = np.minimum(maskt[:,:,1:], maskt[:,:,:-1])
maskv[:,1:,:] = np.minimum(maskt[:,1:,:], maskt[:,:-1,:])
#masku[:,:,1:] *= maskt[:,:,:-1]
#masku[:,:,:-1] *= maskt[:,:,1:]
#maskv[:,1:,:] *= maskt[:,:-1,:]
#maskv[:,:-1,:] *= maskt[:,1:,:]
masku = bnd_exch(masku)
maskv = bnd_exch(maskv)

Tho_dif = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
Tho_adv = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
Tho_mix = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
Tho_con = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
Tho_for = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))

Tuo_dif = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
Tvo_dif = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
Tuo_adv = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
Tvo_adv = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
Tuo_cor = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
Tvo_cor = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
Tuo_pgr = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
Tvo_pgr = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
Tuo_for = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
Tvo_for = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
Tuo_vdf = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
Tvo_vdf = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
Tuo_for = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
Tvo_for = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))

ft = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
fu = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
fv = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
ft[:,nspy:-nspy,nspx:-nspx] = ft0
fu[:,nspy:-nspy,nspx:-nspx] = fu0
fv[:,nspy:-nspy,nspx:-nspx] = fv0
ft = bnd_exch(ft)
fu = bnd_exch(fu)
fv = bnd_exch(fv)
ft *= maskt
fu *= masku
fv *= maskv
a_coru = 1./(dt**2*fu**2+1.)
a_corv = 1./(dt**2*fv**2+1.)
b_coru = dt*fu/(dt**2*fu**2+1.)
b_corv = dt*fv/(dt**2*fv**2+1.)

eta = np.ma.zeros((nz+1,ny+2*nspy,nx+2*nspx))
eta[:,nspy:-nspy,nspx:-nspx] = eta0
eta *= maskt[0:1,:,:]
eta = bnd_exch(eta)

uo0 *= masku[:,nspy:-nspy,nspx:-nspx]
vo0 *= maskv[:,nspy:-nspy,nspx:-nspx]
ho0 *= maskt[:,nspy:-nspy,nspx:-nspx]
uo = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
vo = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
ho = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
uo[:,nspy:-nspy,nspx:-nspx] = uo0
vo[:,nspy:-nspy,nspx:-nspx] = vo0
ho[:,nspy:-nspy,nspx:-nspx] = ho0
uo *= masku
vo *= maskv
ho *= maskt
uo = bnd_exch(uo)
vo = bnd_exch(vo)
ho = bnd_exch(ho)

taux = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
tauy = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
taux[0,nspy:-nspy,nspx:-nspx] = taux0
tauy[0,nspy:-nspy,nspx:-nspx] = tauy0
taux = bnd_exch(taux)
tauy = bnd_exch(tauy)
taux *= masku
tauy *= maskv

if do_height_stommel_arons:
  dw_source = np.ma.zeros((nz,ny+2*nspy,nx+2*nspx))
  dw_source[:,nspy:-nspy,nspx:-nspx] = dw_source0
  dw_source = bnd_exch(dw_source)
  dw_source *= maskt

# Prepare netcdf output
# ---------------------
lev = np.arange(nz)
ds_empty = xr.Dataset(coords=dict(xt=xt, yt=yt, xu=xu, yu=yu, lev=lev))
#dimsu = ['time', 'lev', 'yt', 'xu']
#dimsv = ['time', 'lev', 'yu', 'xt']
#dimst = ['time', 'lev', 'yt', 'xt']
dimsu = ['lev', 'yt', 'xu']
dimsv = ['lev', 'yu', 'xt']
dimst = ['lev', 'yt', 'xt']

# Start time loop
# ---------------
time = 0.
time_wc_ini = datetime.datetime.now()
for ll in range(nt):
  time += dt

  if ll%diagnostic_frequency==0:
    cflu = dt/dx*uo.max()
    cflv = dt/dy*vo.max()
    ke = 0.5*(uo**2+vo**2)
    print(f'll = {ll}/{nt}, ho.sum = {ho.sum()}, ke.sum = {ke.sum()}, cflu = {cflu}, cflv = {cflv}')
  
  # Diffusion
  # ---------
  if do_height_diffusion:
    flux_x = np.zeros((nz,ny+2*nspy,nx+2*nspx))
    flux_y = np.zeros((nz,ny+2*nspy,nx+2*nspx))
    flux_x[:,:,1:] = kh*(ho[:,:,1:]-ho[:,:,:-1])/dx * masku[:,:,1:]
    flux_y[:,1:,:] = kh*(ho[:,1:,:]-ho[:,:-1,:])/dy * maskv[:,1:,:]
    Tho_dif[:,:-1,:-1] = (flux_x[:,:-1,1:]-flux_x[:,:-1,:-1])/dx + (flux_y[:,1:,:-1]-flux_y[:,:-1,:-1])/dy
    #Tho_dif[:,1:-1,1:-1] = (  
    #    kh*(ho[:,2:,1:-1] - 2.*ho[:,1:-1,1:-1] + ho[:,:-2,1:-1])/dx**2
    #  + kh*(ho[:,1:-1,2:] - 2.*ho[:,1:-1,1:-1] + ho[:,1:-1,:-2])/dy**2 
    #)
  if do_momentum_diffusion:
    flux_x = np.zeros((nz,ny+2*nspy,nx+2*nspx))
    flux_y = np.zeros((nz,ny+2*nspy,nx+2*nspx))
    flux_x[:,:,:-1] = Ah*(uo[:,:,1:]-uo[:,:,:-1])/dx * maskt[:,:,:-1]
    flux_y[:,1:,:] = Ah*(uo[:,1:,:]-uo[:,:-1,:])/dy * maskv[:,1:,:]
    Tuo_dif[:,:-1,1:] = (flux_x[:,:-1,1:]-flux_x[:,:-1,:-1])/dx + (flux_y[:,1:,1:]-flux_y[:,:-1,1:])/dy
    Tuo_dif *= masku
    flux_x = np.zeros((nz,ny+2*nspy,nx+2*nspx))
    flux_y = np.zeros((nz,ny+2*nspy,nx+2*nspx))
    flux_x[:,:,1:] = Ah*(vo[:,:,1:]-vo[:,:,:-1])/dx * masku[:,:,1:]
    flux_y[:,:-1,:] = Ah*(vo[:,1:,:]-vo[:,:-1,:])/dy * maskt[:,:-1,:]
    Tvo_dif[:,1:,:-1] = (flux_x[:,1:,1:]-flux_x[:,1:,:-1])/dx + (flux_y[:,1:,:-1]-flux_y[:,:-1,:-1])/dy
    Tvo_dif *= maskv
    #Tuo_dif[:,1:-1,1:-1] = (  
    #    Ah*(uo[:,2:,1:-1] - 2.*uo[:,1:-1,1:-1] + uo[:,:-2,1:-1])/dx**2
    #  + Ah*(uo[:,1:-1,2:] - 2.*uo[:,1:-1,1:-1] + uo[:,1:-1,:-2])/dy**2 
    #)
    #Tvo_dif[:,1:-1,1:-1] = (  
    #    Ah*(vo[:,2:,1:-1] - 2.*vo[:,1:-1,1:-1] + vo[:,:-2,1:-1])/dx**2
    #  + Ah*(vo[:,1:-1,2:] - 2.*vo[:,1:-1,1:-1] + vo[:,1:-1,:-2])/dy**2 
    #)
  if do_momentum_drag:
    Tuo_dif += -drag_coeff_linear*uo
    Tvo_dif += -drag_coeff_linear*vo

  # Advection
  # ---------
  if do_height_advection:
    flux_x = np.zeros((nz,ny+2*nspy,nx+2*nspx))
    flux_y = np.zeros((nz,ny+2*nspy,nx+2*nspx))
    if False:
      # --- 2nd order central
      flux_x[:,:,1:]  = -0.5*(ho[:,:,1:]+ho[:,:,:-1])*uo[:,:,1:] * masku[:,:,1:]
      flux_y[:,1:,:]  = -0.5*(ho[:,1:,:]+ho[:,:-1,:])*vo[:,1:,:] * maskv[:,1:,:]
    else:
      # --- 1st order upwind
      # sig=1: (1+sig)/2*hop1*uop1 + (1-(1+sig)/2)*hom1*uom1
      sig_fac = 0.5*(1.+np.sign(uo))
      flux_x[:,:,1:]  = -(sig_fac[:,:,1:]*ho[:,:,:-1]*uo[:,:,1:] + (1.-sig_fac[:,:,1:])*ho[:,:,1:]*uo[:,:,1:]) * masku[:,:,1:]
      sig_fac = 0.5*(1.+np.sign(vo))
      flux_y[:,1:,:]  = -(sig_fac[:,1:,:]*ho[:,:-1,:]*vo[:,1:,:] + (1.-sig_fac[:,1:,:])*ho[:,1:,:]*vo[:,1:,:]) * maskv[:,1:,:]
    Tho_adv[:,:-1,:-1] = (flux_x[:,:-1,1:]-flux_x[:,:-1,:-1])/dx + (flux_y[:,1:,:-1]-flux_y[:,:-1,:-1])/dy
    #Tho_adv[:,1:-1,1:-1] = -(
    #    ( 0.5*(ho[:,1:-1,2:]+ho[:,1:-1,1:-1])*uo[:,1:-1,2:] - 0.5*(ho[:,1:-1,1:-1]+ho[:,1:-1,:-2])*uo[:,1:-1,1:-1] ) / dx
    #  + ( 0.5*(ho[:,2:,1:-1]+ho[:,1:-1,1:-1])*vo[:,2:,1:-1] - 0.5*(ho[:,1:-1,1:-1]+ho[:,:-2,1:-1])*vo[:,1:-1,1:-1] ) / dy
    #)
  if do_momentum_advection:
    Tuo_adv[:,1:-1,1:-1] = -(
        ( 0.5*(uo[:,1:-1,2:]+uo[:,1:-1,1:-1])*0.5*(uo[:,1:-1,2:]+uo[:,1:-1,1:-1]) - 0.5*(uo[:,1:-1,1:-1]+uo[:,1:-1,:-2])*0.5*(uo[:,1:-1,1:-1]+uo[:,1:-1,:-2]) ) / dx
      + ( 0.5*(uo[:,2:,1:-1]+uo[:,1:-1,1:-1])*0.5*(vo[:,2:,1:-1]+vo[:,2:,:-2]) - 0.5*(uo[:,1:-1,1:-1]+uo[:,:-2,1:-1])*0.5*(vo[:,1:-1,1:-1]+vo[:,1:-1,:-2]) ) / dy
    )
    Tvo_adv[:,1:-1,1:-1] = -(
        ( 0.5*(vo[:,1:-1,2:]+vo[:,1:-1,1:-1])*0.5*(uo[:,1:-1,2:]+uo[:,:-2,2:]) - 0.5*(vo[:,1:-1,1:-1]+vo[:,1:-1,:-2])*0.5*(uo[:,1:-1,1:-1]+uo[:,:-2,1:-1]) ) / dx
      + ( 0.5*(vo[:,2:,1:-1]+vo[:,1:-1,1:-1])*0.5*(vo[:,2:,1:-1]+vo[:,1:-1,1:-1]) - 0.5*(vo[:,1:-1,1:-1]+vo[:,:-2,1:-1])*0.5*(vo[:,1:-1,1:-1]+vo[:,:-2,1:-1]) ) / dy
    )

  # Coriolis
  # --------
  if do_momentum_coriolis_exp:
    Tuo_cor[:,:-1,1:] =  0.5*( ft[:,:-1,1:]*0.5*(vo[:,1:,1:]+vo[:,:-1,1:]) + ft[:,:-1,:-1]*0.5*(vo[:,1:,:-1]+vo[:,:-1,:-1]) ) * masku[:,:-1,1:]
    Tvo_cor[:,1:,:-1] = -0.5*( ft[:,1:,:-1]*0.5*(uo[:,1:,1:]+uo[:,1:,:-1]) + ft[:,:-1,:-1]*0.5*(uo[:,:-1,1:]+uo[:,:-1,:-1]) ) * maskv[:,:-1,1:]

  # Pressure gradient
  # -----------------
  if do_momentum_pressure_gradient:
    eta[:nz,:,:] = eta[nz,:,:] + np.cumsum(ho[::-1,:,:],axis=0)[::-1,:,:]
    pres = np.cumsum(gprime*eta[:-1,:,:], axis=0)
    Tuo_pgr[:,:,1:] = -(pres[:,:,1:]-pres[:,:,:-1])/dx * masku[:,:,1:]
    Tvo_pgr[:,1:,:] = -(pres[:,1:,:]-pres[:,:-1,:])/dy * maskv[:,1:,:]

  # Momentum forcing
  # ----------------
  if do_momentum_windstress:
    Tuo_vdf[0,:,:] = taux[0,:,:]/(ho[0,:,:]+1e-33)
    Tvo_vdf[0,:,:] = tauy[0,:,:]/(ho[0,:,:]+1e-33)

  if do_momentum_forcing:
    Tuo_for = uo_forcing(time)
    Tvo_for = vo_forcing(time)

  # Height sources / sinks
  # ----------------------
  if do_height_stommel_arons:
    Tho_mix[:] = -lam_ho*ho
    Tho_con[:] = dw_source 
  if do_height_forcing:
    Tho_for = ho_forcing(time)

  # Sum of tendencies
  # -----------------
  Tho_tot = Tho_dif + Tho_adv + Tho_mix + Tho_con + Tho_for
  Tuo_tot = Tuo_dif + Tuo_adv + Tuo_cor + Tuo_pgr + Tuo_for + Tuo_vdf
  Tvo_tot = Tvo_dif + Tvo_adv + Tvo_cor + Tvo_pgr + Tvo_for + Tvo_vdf

  Tho_tot = bnd_exch(Tho_tot)
  Tuo_tot = bnd_exch(Tuo_tot)
  Tvo_tot = bnd_exch(Tvo_tot)
  Tho_tot *= maskt
  Tuo_tot *= masku
  Tvo_tot *= maskv

  # AB time stepping
  # ----------------
  # Use Euler step for ll==0: old tend = new tend
  if ll==0:
    Tho_tot_old = Tho_tot
    Tuo_tot_old = Tuo_tot
    Tvo_tot_old = Tvo_tot

  # time step
  ho = ho + dt*( (1.5+epsab)*Tho_tot - (0.5+epsab)*Tho_tot_old )
  uo = uo + dt*( (1.5+epsab)*Tuo_tot - (0.5+epsab)*Tuo_tot_old )
  vo = vo + dt*( (1.5+epsab)*Tvo_tot - (0.5+epsab)*Tvo_tot_old )
  if do_momentum_coriolis_imp:
    uo_old = uo.copy()
    vo_old = vo.copy()
    #uo =  a_coru * uo_old + b_corv * vo_old
    #vo = -b_coru * uo_old + a_corv * vo_old
    uo[:,:-1,1:] =  a_coru[:,:-1,1:] * uo_old[:,:-1,1:] + b_coru[:,:-1,1:] * 0.25*(vo_old[:,:-1,:-1]+vo_old[:,:-1,1:]+vo_old[:,1:,1:]+vo_old[:,1:,:-1])
    vo[:,1:,:-1] =  a_corv[:,1:,:-1] * vo_old[:,1:,:-1] - b_corv[:,1:,:-1] * 0.25*(uo_old[:,:-1,:-1]+uo_old[:,:-1,1:]+uo_old[:,1:,1:]+uo_old[:,1:,:-1])
    uo *= masku
    vo *= maskv
    Tuo_cor = (uo-uo_old)/dt
    Tvo_cor = (vo-vo_old)/dt
    Tuo_tot += Tuo_cor
    Tvo_tot += Tvo_cor

  # boundary exchange
  ho = bnd_exch(ho)
  uo = bnd_exch(uo)
  vo = bnd_exch(vo)

  # update tendencies for AB
  Tho_tot_old = Tho_tot
  Tuo_tot_old = Tuo_tot
  Tvo_tot_old = Tvo_tot

  # Diagnostics
  # -----------
  times[ll] = time 
  uo_ts[ll,:] = uo[0,iy,ix]
  vo_ts[ll,:] = vo[0,iy,ix]
  ho_ts[ll,:] = ho[0,iy,ix]

  # Output variables
  # ----------------
  if output_frequency!=0 and ll%output_frequency==0:
    nnp+=1 

    ds = ds_empty.copy()
    ds['time'] = time
    ds['uo'] = create_output_var(uo, dimsu)
    ds['vo'] = create_output_var(vo, dimsv)
    ds['ho'] = create_output_var(ho, dimst)
    fpath = f'{path_data}/{file_prfx}_{nnp:04d}.nc'
    print(f'Save file {fpath}')
    ds.to_netcdf(fpath)

    if nnp!=1:
      dt_wc, time_wc_fin = timing(time_wc_ini, verbose=True)


  # Plot variables
  # --------------
  if picture_frequency!=0 and ll%picture_frequency==0:
    nnf+=1

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

    plt.close('all')

    # --- mom. tend.
    hca, hcb = pyic.arrange_axes(5,3, plot_cb=True, asp=1., fig_size_fac=1.)
    ii=-1
    
    clim = 'sym'
    
    ii+=1; ax=hca[ii]; cax=hcb[ii]
    pyic.shade(xt/1e3, yt/1e3, uop[0,:,:], ax=ax, cax=cax, clim=clim)
    #pyic.shade(xt/1e3, yt/1e3, vop[0,:,:], ax=ax, cax=cax, clim=clim)
    ax.set_title(f'uo')
    
    ii+=1; ax=hca[ii]; cax=hcb[ii]
    pyic.shade(xt/1e3, yt/1e3, hop[0,:,:], ax=ax, cax=cax, clim=clim)
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

    fpath = f'{path_data}/{fig_prfx}_{nnf:04d}.png'
    print(f'Save figure {fpath}')
    plt.savefig(fpath)

  # Write restart
  # -------------

print('--- All done! ---')
time_wc_now = datetime.datetime.now()
dt_wc = (time_wc_now - time_wc_ini)
print(f'Total run time: {dt_wc.total_seconds()/60:.1f}min, done at {time_wc_now}')
print('------')
