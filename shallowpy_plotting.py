""" The essential plotting routines are taken from pyicon.
(https://gitlab.dkrz.de/m300602/pyicon)
"""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.collections import PolyCollection
import numpy as np
import sys
import cmocean

def arrange_axes(nx,ny,
                 sharex = True,
                 sharey = False,
                 xlabel = '',
                 ylabel = '',
                 # labeling axes with e.g. (a), (b), (c)
                 do_axes_labels = True,
                 axlab_kw = dict(),
                 # colorbar
                 plot_cb = True,
                 # projection (e.g. for cartopy)
                 projection = None,
                 # aspect ratio of axes
                 asp = 1.,
                 sasp = 0.,  # for compability with older version of arrange_axes
                 # width and height of axes
                 wax = 'auto',
                 hax = 4.,
                 # extra figure spaces (left, right, top, bottom)
                 dfigl = 0.0,
                 dfigr = 0.0,
                 dfigt = 0.0,
                 dfigb = 0.0,
                 # space aroung axes (left, right, top, bottom) 
                 daxl = 1.8, # reset to zero if sharex==False
                 daxr = 0.8,
                 daxt = 0.8,
                 daxb = 1.2, # reset to zero if sharex==True
                 # space around colorbars (left, right, top, bottom) 
                 dcbl = -0.5,
                 dcbr = 1.4,
                 dcbt = 0.0,
                 dcbb = 0.5,
                 # width and height of colorbars
                 wcb = 0.5,
                 hcb = 'auto',
                 # factors to increase widths and heights of axes and colorbars
                 fig_size_fac = 1.,
                 f_wax = 1.,
                 f_hax = 1.,
                 f_wcb = 1.,
                 f_hcb = 1.,
                 # factors to increase spaces (figure)
                 f_dfigl = 1.,
                 f_dfigr = 1.,
                 f_dfigt = 1.,
                 f_dfigb = 1.,
                 # factors to increase spaces (axes)
                 f_daxl = 1.,
                 f_daxr = 1.,
                 f_daxt = 1.,
                 f_daxb = 1.,
                 # factors to increase spaces (colorbars)
                 f_dcbl = 1.,
                 f_dcbr = 1.,
                 f_dcbt = 1.,
                 f_dcbb = 1.,
                 # font sizes of labels, titles, ticks
                 fs_label = 10.,
                 fs_title = 12.,
                 fs_ticks = 10.,
                 # font size increasing factor
                 f_fs = 1,
                 reverse_order = False,
                ):

  # factor to convert cm into inch
  cm2inch = 0.3937

  if sasp!=0:
    print('::: Warning: You are using keyword ``sasp`` for setting the aspect ratio but you should switch to use ``asp`` instead.:::')
    asp = 1.*sasp

  # --- set hcb in case it is auto
  if isinstance(wax, str) and wax=='auto':
    wax = hax/asp

  # --- set hcb in case it is auto
  if isinstance(hcb, str) and hcb=='auto':
    hcb = hax

  # --- rename horizontal->bottom and vertical->right
  if isinstance(plot_cb, str) and plot_cb=='horizontal':
    plot_cb = 'bottom'
  if isinstance(plot_cb, str) and plot_cb=='vertical':
    plot_cb = 'right'
  
  # --- apply fig_size_fac
  # font sizes
  #f_fs *= fig_size_fac
  # factors to increase widths and heights of axes and colorbars
  f_wax *= fig_size_fac
  f_hax *= fig_size_fac
  #f_wcb *= fig_size_fac
  f_hcb *= fig_size_fac
  ## factors to increase spaces (figure)
  #f_dfigl *= fig_size_fac
  #f_dfigr *= fig_size_fac
  #f_dfigt *= fig_size_fac
  #f_dfigb *= fig_size_fac
  ## factors to increase spaces (axes)
  #f_daxl *= fig_size_fac
  #f_daxr *= fig_size_fac
  #f_daxt *= fig_size_fac
  #f_daxb *= fig_size_fac
  ## factors to increase spaces (colorbars)
  #f_dcbl *= fig_size_fac
  #f_dcbr *= fig_size_fac
  #f_dcbt *= fig_size_fac
  #f_dcbb *= fig_size_fac
  
  # --- apply font size factor
  fs_label *= f_fs
  fs_title *= f_fs
  fs_ticks *= f_fs

  # make vector of plot_cb if it has been true or false before
  # plot_cb can have values [{1}, 0] 
  # with meanings:
  #   1: plot cb; 
  #   0: do not plot cb
  plot_cb_right  = False
  plot_cb_bottom = False
  if isinstance(plot_cb, bool) and (plot_cb==True):
    plot_cb = np.ones((nx,ny))  
  elif isinstance(plot_cb, bool) and (plot_cb==False):
    plot_cb = np.zeros((nx,ny))
  elif isinstance(plot_cb, str) and plot_cb=='right':
    plot_cb = np.zeros((nx,ny))
    plot_cb_right = True
  elif isinstance(plot_cb, str) and plot_cb=='bottom':
    plot_cb = np.zeros((nx,ny))
    plot_cb_bottom = True
  else:
    plot_cb = np.array(plot_cb)
    if plot_cb.size!=nx*ny:
      raise ValueError('Vector plot_cb has wrong length!')
    if plot_cb.shape[0]==nx*ny:
      plot_cb = plot_cb.reshape(ny,nx).transpose()
    elif plot_cb.shape[0]==ny:
      plot_cb = plot_cb.transpose()
  
  # --- make list of projections if it is not a list
  if not isinstance(projection, list):
    projection = [projection]*nx*ny
  
  # --- make arrays and multiply by f_*
  daxl = np.array([daxl]*nx)*f_daxl
  daxr = np.array([daxr]*nx)*f_daxr
  dcbl = np.array([dcbl]*nx)*f_dcbl
  dcbr = np.array([dcbr]*nx)*f_dcbr
  
  wax = np.array([wax]*nx)*f_wax
  wcb = np.array([wcb]*nx)*f_wcb
  
  daxt = np.array([daxt]*ny)*f_daxt
  daxb = np.array([daxb]*ny)*f_daxb
  dcbt = np.array([dcbt]*ny)*f_dcbt
  dcbb = np.array([dcbb]*ny)*f_dcbb
  
  hax = np.array([hax]*ny)*f_hax
  hcb = np.array([hcb]*ny)*f_hcb
  
  # --- adjust for shared axes
  if sharex:
    daxb[:-1] = 0.
  
  if sharey:
    daxl[1:] = 0.

  # --- adjust for one colorbar at the right or bottom
  if plot_cb_right:
    daxr_s = daxr[0]
    dcbl_s = dcbl[0]
    dcbr_s = dcbr[0]
    wcb_s  = wcb[0]
    hcb_s  = hcb[0]
    dfigr += dcbl_s+wcb_s+0.*dcbr_s+daxl[0]
  if plot_cb_bottom:
    hcb_s  = wcb[0]
    wcb_s  = wax[0]
    dcbb_s = dcbb[0]+daxb[-1]
    dcbt_s = dcbt[0]
    #hcb_s  = hcb[0]
    dfigb += dcbb_s+hcb_s+dcbt_s
  
  # --- adjust for columns without colorbar
  delete_cb_space = plot_cb.sum(axis=1)==0
  dcbl[delete_cb_space] = 0.0
  dcbr[delete_cb_space] = 0.0
  wcb[delete_cb_space]  = 0.0
  
  # --- determine ax position and fig dimensions
  x0 =   dfigl
  y0 = -(dfigt)
  
  pos_axcm = np.zeros((nx*ny,4))
  pos_cbcm = np.zeros((nx*ny,4))
  nn = -1
  y00 = y0
  x00 = x0
  for jj in range(ny):
    y0   += -(daxt[jj]+hax[jj])
    x0 = x00
    for ii in range(nx):
      nn += 1
      x0   += daxl[ii]
      pos_axcm[nn,:] = [x0, y0, wax[ii], hax[jj]]
      pos_cbcm[nn,:] = [x0+wax[ii]+daxr[ii]+dcbl[ii], y0, wcb[ii], hcb[jj]]
      x0   += wax[ii]+daxr[ii]+dcbl[ii]+wcb[ii]+dcbr[ii]
    y0   += -(daxb[jj])
  wfig = x0+dfigr
  hfig = y0-dfigb
  
  # --- transform from negative y axis to positive y axis
  hfig = -hfig
  pos_axcm[:,1] += hfig
  pos_cbcm[:,1] += hfig
  
  # --- convert to fig coords
  cm2fig_x = 1./wfig
  cm2fig_y = 1./hfig
  
  pos_ax = 1.*pos_axcm
  pos_cb = 1.*pos_cbcm
  
  pos_ax[:,0] = pos_axcm[:,0]*cm2fig_x
  pos_ax[:,2] = pos_axcm[:,2]*cm2fig_x
  pos_ax[:,1] = pos_axcm[:,1]*cm2fig_y
  pos_ax[:,3] = pos_axcm[:,3]*cm2fig_y
  
  pos_cb[:,0] = pos_cbcm[:,0]*cm2fig_x
  pos_cb[:,2] = pos_cbcm[:,2]*cm2fig_x
  pos_cb[:,1] = pos_cbcm[:,1]*cm2fig_y
  pos_cb[:,3] = pos_cbcm[:,3]*cm2fig_y

  # --- find axes center (!= figure center)
  x_ax_cent = pos_axcm[0,0] +0.5*(pos_axcm[-1,0]+pos_axcm[-1,2]-pos_axcm[0,0])
  y_ax_cent = pos_axcm[-1,1]+0.5*(pos_axcm[0,1] +pos_axcm[0,3] -pos_axcm[-1,1])
  
  # --- make figure and axes
  fig = plt.figure(figsize=(wfig*cm2inch, hfig*cm2inch))
  
  hca = [0]*(nx*ny)
  hcb = [0]*(nx*ny)
  nn = -1
  for jj in range(ny):
    for ii in range(nx):
      nn+=1
  
      # --- axes
      hca[nn] = fig.add_subplot(position=pos_ax[nn,:], projection=projection[nn])
      hca[nn].set_position(pos_ax[nn,:])
  
      # --- colorbar
      if plot_cb[ii,jj] == 1:
        hcb[nn] = fig.add_subplot(position=pos_cb[nn,:])
        hcb[nn].set_position(pos_cb[nn,:])
      ax  = hca[nn]
      cax = hcb[nn] 
  
      # --- label
      ax.set_xlabel(xlabel, fontsize=fs_label)
      ax.set_ylabel(ylabel, fontsize=fs_label)
      #ax.set_title('', fontsize=fs_title)
      matplotlib.rcParams['axes.titlesize'] = fs_title
      ax.tick_params(labelsize=fs_ticks)
      if plot_cb[ii,jj] == 1:
        hcb[nn].tick_params(labelsize=fs_ticks)
  
      #ax.tick_params(pad=-10.0)
      #ax.xaxis.labelpad = 0
      #ax._set_title_offset_trans(float(-20))
  
      # --- axes ticks
      # delete labels for shared axes
      if sharex and jj!=ny-1:
        hca[nn].ticklabel_format(axis='x',style='plain',useOffset=False)
        hca[nn].tick_params(labelbottom=False)
        hca[nn].set_xlabel('')
  
      if sharey and ii!=0:
        hca[nn].ticklabel_format(axis='y',style='plain',useOffset=False)
        hca[nn].tick_params(labelleft=False)
        hca[nn].set_ylabel('')
  
      # ticks for colorbar 
      if plot_cb[ii,jj] == 1:
        hcb[nn].set_xticks([])
        hcb[nn].yaxis.tick_right()
        hcb[nn].yaxis.set_label_position("right")

  #--- needs to converted to fig coords (not cm)
  if plot_cb_right:
    nn = -1
    #pos_cb = np.array([(wfig-(dfigr+dcbr_s+wcb_s))*cm2fig_x, (y_ax_cent-0.5*hcb_s)*cm2fig_y, wcb_s*cm2fig_x, hcb_s*cm2fig_y])
    pos_cb = np.array([ (pos_axcm[-1,0]+pos_axcm[-1,2]+daxr_s+dcbl_s)*cm2fig_x, 
                        (y_ax_cent-0.5*hcb_s)*cm2fig_y, 
                        (wcb_s)*cm2fig_x, 
                        (hcb_s)*cm2fig_y 
                      ])
    hcb[nn] = fig.add_subplot(position=pos_cb)
    hcb[nn].tick_params(labelsize=fs_ticks)
    hcb[nn].set_position(pos_cb)
    hcb[nn].set_xticks([])
    hcb[nn].yaxis.tick_right()
    hcb[nn].yaxis.set_label_position("right")

  if plot_cb_bottom:
    nn = -1
    pos_cb = np.array([ (x_ax_cent-0.5*wcb_s)*cm2fig_x, 
                        (dcbb_s)*cm2fig_y, 
                        (wcb_s)*cm2fig_x, 
                        (hcb_s)*cm2fig_y
                      ])
    hcb[nn] = fig.add_subplot(position=pos_cb)
    hcb[nn].set_position(pos_cb)
    hcb[nn].tick_params(labelsize=fs_ticks)
    hcb[nn].set_yticks([])

  if reverse_order:
    isort = np.arange(nx*ny, dtype=int).reshape((ny,nx)).transpose().flatten()
    hca = list(np.array(hca)[isort]) 
    hcb = list(np.array(hcb)[isort])

  # add letters for subplots
  if (do_axes_labels) and (axlab_kw is not None):
    hca = axlab(hca, fontdict=axlab_kw)

  return hca, hcb

# ================================================================================ 
def axlab(hca, figstr=[], posx=[-0.00], posy=[1.05], fontdict=None):
  """
input:
----------
  hca:      list with axes handles
  figstr:   list with strings that label the subplots
  posx:     list with length 1 or len(hca) that gives the x-coordinate in ax-space
  posy:     list with length 1 or len(hca) that gives the y-coordinate in ax-space
last change:
----------
2015-07-21
  """

  # make list that looks like [ '(a)', '(b)', '(c)', ... ]
  if len(figstr)==0:
    #lett = "abcdefghijklmnopqrstuvwxyz"
    lett  = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
    lett += ["a2","b2","c2","d2","e2","f2","g2","h2","i2","j2","k2","l2","m2","n2","o2","p2","q2","r2","s2","t2","u2","v2","w2","x2","y2","z2"]
    lett = lett[0:len(hca)]
    figstr = ["z"]*len(hca)
    for nn, ax in enumerate(hca):
      figstr[nn] = "(%s)" % (lett[nn])
  
  if len(posx)==1:
    posx = posx*len(hca)
  if len(posy)==1:
    posy = posy*len(hca)
  
  # draw text
  for nn, ax in enumerate(hca):
    ht = hca[nn].text(posx[nn], posy[nn], figstr[nn], 
                      transform = hca[nn].transAxes, 
                      horizontalalignment = 'right',
                      fontdict=fontdict)
    # add text handle to axes to give possibility of changing text properties later
    # e.g. by hca[nn].axlab.set_fontsize(8)
    hca[nn].axlab = ht
#  for nn, ax in enumerate(hca):
#    #ax.set_title(figstr[nn]+'\n', loc='left', fontsize=10)
#    ax.set_title(figstr[nn], loc='left', fontsize=10)
  return hca

def shade(
              x='auto', y='auto', datai='auto',
              ax='auto', cax=0,
              cmap='auto',
              cincr=-1.,
              norm=None,
              rasterized=True,
              clim=[None, None],
              extend='both',
              clevs=None,
              contfs=None,
              conts=None,
              nclev='auto',
              #cint='auto', # old: use cincr now
              contcolor='k',
              contthick=0.,
              contlw=1.,
              use_pcol=True,
              use_pcol_or_contf=True,
              cbticks='auto',
              cbtitle='',
              cbdrawedges='auto',
              #cborientation='vertical',
              cborientation='auto',
              cbkwargs=None,
              adjust_axlims=True,
              bmp=None,
              transform=None,
              projection=None,
              logplot=False,
              edgecolor='none',
           ):
    """ Convenient wrapper around pcolormesh, contourf, contour and their triangular versions.
    """
    # --- decide whether regular or triangular plots should be made
    if isinstance(datai, str) and datai=='auto':
      Tri = x
      datai = y
      rectangular_grid = False
    else:
      rectangular_grid = True

    if projection is not None:
      transform = projection


    # --- decide whether pcolormesh or contourf plot
    if use_pcol_or_contf:
      if contfs is None:
        use_pcol = True
        use_contf = False
      else:
        use_pcol = False
        use_contf = True
    else:
        use_pcol = False
        use_contf = False
    #if use_pcol and use_contf:
    #  raise ValueError('::: Error: Only one of use_pcol or use_contf can be True. :::')

    # --- mask 0 and negative values in case of log plot
    #data = 1.*datai
    data = datai.copy()
    if logplot and isinstance(data, np.ma.MaskedArray):
      data[data<=0.0] = np.ma.masked
      data = np.ma.log10(data) 
    elif logplot and not isinstance(data, np.ma.MaskedArray):
      data[data<=0.0] = np.nan
      data = np.log10(data) 
  
    # --- clim
    if isinstance(clim, str) and clim=='auto':
      clim = [None, None]
    elif isinstance(clim, str) and clim=='sym':
      clim = np.abs(data).max()
    clim=np.array(clim)
    if clim.size==1:
      clim = np.array([-1, 1])*clim
    if clim[0] is None:
      clim[0] = data.min()
    if clim[1] is None:
      clim[1] = data.max()
  
    # --- cmap
    if (clim[0]==-clim[1]) and cmap=='auto':
      cmap = 'RdBu_r'
    elif cmap=='auto':
      #cmap = 'viridis'
      cmap = 'RdYlBu_r'
    if isinstance(cmap, str):
      cmap = getattr(plt.cm, cmap)
  
    if use_pcol:
      # --- norm
      if cincr>0.:
        clevs = np.arange(clim[0], clim[1]+cincr, cincr)
        use_norm = True
      elif use_pcol and clevs is not None:
        clevs = np.array(clevs)
        use_norm = True
      elif norm is not None:
        use_norm = False # prevent that norm is overwritten later on
      else:
        norm = None
        use_norm = False
    elif use_contf:
      contfs = calc_conts(contfs, clim, cincr, nclev)
      clevs = contfs
      use_norm = True
    else:
      use_norm = False

    if use_norm:
      #norm = matplotlib.colors.BoundaryNorm(boundaries=clevs, ncolors=cmap.N)
      nlev = clevs.size
      # --- expanded norm and cmap
      norm_e = matplotlib.colors.BoundaryNorm(boundaries=np.arange(0,nlev+2,1), ncolors=cmap.N)
      cmap_e = matplotlib.colors.ListedColormap(cmap(norm_e(np.arange(0,nlev+1,1))))
      # --- actuall cmap with over and under values
      cmap = matplotlib.colors.ListedColormap(cmap(norm_e(np.arange(1,nlev,1))))        
      norm = matplotlib.colors.BoundaryNorm(boundaries=clevs, ncolors=cmap.N)
      cmap.set_under(cmap_e(norm_e(0)))
      cmap.set_over(cmap_e(norm_e(nlev)))
      vmin = None
      vmax = None
    elif norm:
      vmin = None
      vmax = None
      clim = [None, None]
    else:
      vmin = clim[0]
      vmax = clim[1]
  
    # --- decide whether to use extra contour lines
    if conts is None:
      use_cont = False
    else:
      use_cont = True
      conts = calc_conts(conts, clim, cincr, nclev)
    if use_norm:
      clim = [None, None]
  
    # --- decide whether there should be black edges at colorbar
    if isinstance(cbdrawedges, str) and cbdrawedges=='auto':
      if use_norm or use_contf:
        cbdrawedges = True
      else:
        cbdrawedges = False
    else:
      cbdrawedges = False
  
    # --- necessary cartopy settings
    ccrsdict = dict()
    if transform is not None:
      ccrsdict = dict(transform=transform)
      #adjust_axlims = False
      #adjust_axlims = True
    
    # --- make axes if necessary
    if ax == 'auto':
      ax = plt.gca()
  
    if rectangular_grid:
      # --- adjust x and y if necessary
      # ------ make x and y 2D
      if x.ndim==1:
        x, y = np.meshgrid(x, y)
  
      # ------ convert to Basemap maps coordinates
      if bmp is not None:
        x, y = bmp(x, y)
        
      # ------ bring x and y to correct shape for contour
      if (use_cont) or (use_contf):
        if x.shape[1] != data.shape[1]:
          xc = 0.25*(x[1:,1:]+x[:-1,1:]+x[1:,:-1]+x[:-1,:-1])
          yc = 0.25*(y[1:,1:]+y[:-1,1:]+y[1:,:-1]+y[:-1,:-1])
        else:
          xc = x.copy()
          yc = y.copy()
      
    # --- allocate list of all plot handles
    hs = []
  
    # --- color plot
    # either pcolormesh plot
    if use_pcol:
      if rectangular_grid:
        hm = ax.pcolormesh(x, y, 
                           data, 
                           vmin=clim[0], vmax=clim[1],
                           cmap=cmap, 
                           norm=norm,
                           rasterized=rasterized,
                           edgecolor=edgecolor,
                           shading='auto',
                           **ccrsdict
                          )
      else:
        hm = ax.tripcolor(Tri, 
                          data, 
                          vmin=clim[0], vmax=clim[1],
                          cmap=cmap, 
                          norm=norm,
                          rasterized=rasterized,
                          edgecolor=edgecolor,
                          **ccrsdict
                         )
      hs.append(hm)
    # or contourf plot
    elif use_contf:
      if rectangular_grid:
        hm = ax.contourf(xc, yc, 
                         data, contfs,
                         vmin=clim[0], vmax=clim[1],
                         cmap=cmap, 
                         norm=norm,
                         extend=extend,
                         **ccrsdict
                        )
      else:
        raise ValueError("::: Error: Triangular contourf not supported yet. :::")
        # !!! This does not work sinc Tri.x.size!=data.size which is natural for the picon Triangulation. Not sure why matplotlib tries to enforce this.
        #hm = ax.tricontourf(Tri,
        #                 data, contfs,
        #                 vmin=clim[0], vmax=clim[1],
        #                 cmap=cmap, 
        #                 norm=norm,
        #                 extend=extend,
        #                 **ccrsdict
        #                   )
      hs.append(hm)
  
      # this prevents white lines if fig is saved as pdf
      for cl in hm.collections: 
        cl.set_edgecolor("face")
        cl.set_rasterized(True)
      # rasterize
      if rasterized:
        zorder = -5
        ax.set_rasterization_zorder(zorder)
        for cl in hm.collections:
# This line causes problems with cartopy and contourfs. The plot seems to be unvisible.
#          cl.set_zorder(zorder - 1)
          cl.set_rasterized(True)
    else:
      hm = None
  
    # --- contour plot (can be in addition to color plot above)
    if use_cont:
      if rectangular_grid:
        hc = ax.contour(xc, yc, data, conts, 
                        colors=contcolor, linewidths=contlw, **ccrsdict)
      else:
        raise ValueError("::: Error: Triangular contour not supported yet. :::")
      # ------ if there is a contour matching contthick it will be made thicker
      try:
        i0 = np.where(hc.levels==contthick)[0][0]
        hc.collections[i0].set_linewidth(2.5*contlw)
      except:
        pass
      hs.append(hc)
  
    # --- colorbar
    if (cax!=0) and (hm is not None): 
      # ------ axes for colorbar needs to be created
      if cax == 1:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="10%", pad=0.1)
      # ------ determine cborientation
      if cborientation=='auto':
        if cax.get_xticks().size==0:
          cborientation = 'vertical'
        else:
          cborientation = 'horizontal'
      if not cbkwargs:
        cbkwargs = dict(orientation=cborientation, extend='both')
      # ------ make actual colorbar
      #cb = plt.colorbar(mappable=hm, cax=cax, orientation=cborientation, extend='both')
      cb = plt.colorbar(mappable=hm, cax=cax, **cbkwargs)
      # ------ prevent white lines if fig is saved as pdf
      cb.solids.set_edgecolor("face")
      # ------ use exponential notation for large colorbar ticks
      try:
        cb.formatter.set_powerlimits((-3, 3))
      except:
        pass
      # ------ colorbar ticks
      if isinstance(cbticks, np.ndarray) or isinstance(cbticks, list):
        cb.set_ticks(cbticks)
      else:
        if use_norm:
          dcl = np.diff(clevs)
          if (np.isclose(dcl, dcl[0])).all(): 
            cb.set_ticks(clevs[::2])
          else:
            cb.set_ticks(clevs)
        elif use_norm==False and norm is not None:
          pass
        else:
          cb.locator = ticker.MaxNLocator(nbins=5)
      cb.update_ticks()
      # ------ colorbar title
      cax.set_title(cbtitle)
      # ------ add cb to list of handles
      hs.append(cb)
  
    # --- axes labels and ticks
    if adjust_axlims:
      ax.locator_params(nbins=5)
      if rectangular_grid: 
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
      else:
        ax.set_xlim(Tri.x.min(), Tri.x.max())
        ax.set_ylim(Tri.y.min(), Tri.y.max())
    return hs 

def calc_conts(conts, clim, cincr, nclev):
  # ------ decide how to determine contour levels
  if isinstance(conts, np.ndarray) or isinstance(conts, list):
    # use given contours 
    conts = np.array(conts)
  else:
    # calculate contours
    # ------ decide whether contours should be calculated by cincr or nclev
    if cincr>0:
      conts = np.arange(clim[0], clim[1]+cincr, cincr)
    else:
      if isinstance(nclev,str) and nclev=='auto':
        nclev = 11
      conts = np.linspace(clim[0], clim[1], nclev)
  return conts
