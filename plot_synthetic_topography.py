import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from hole_function_defs import *

"""
This is the code to generate Figs. 1c and d of the paper "Critical percolation threshold is an upper bound on Arctic sea ice melt pond coverage." 
"""

N = 1                          # number of holes to open

res = 500                       # size of the domain
mode = 'snow_dune'              # topography type
tmax = 2; dt = 0.1              # diffusion time and time-step if mode = 'diffusion' or mode = 'rayleigh'
g = 1                           # anisotropy parameter
sigma_h = 0.03                  # surface standard deviation
snow_dune_radius = 1.           # mean snow dune radius if mode = 'snow_dune'  
Gaussians_per_pixel = 0.2       # density of snow dunes if mode = 'snow_dune'  
snow_dune_height_exponent = 1.  # exponent that relates snow dune radius and snow dune height if mode = 'snow_dune' 

Tdrain = 10.; dt_drain = 0.5    # time and time-step of to drainage

# create topography
ice_topo = Create_Initial_Topography(res = res, mode = mode, tmax = tmax, dt = dt, g = g, sigma_h = sigma_h, h = 0., snow_dune_radius = snow_dune_radius, 
            Gaussians_per_pixel = Gaussians_per_pixel, number_of_r_bins = 150, window_size = 5, snow_dune_height_exponent = snow_dune_height_exponent)

# initial water height above all topography
w0 = np.max(ice_topo) + 1E-5                 
water_level = w0 * np.ones([res,res])                                
ponds = (ice_topo < w0).astype(np.uint8)

# open holes randomly
holes = np.random.randint(0,res,size = (N, 2))

# define depth drained during Tdrain
DH_drain = np.max(water_level[holes[:,0],holes[:,1]] - ice_topo[holes[:,0],holes[:,1]])

# drain ponds
ice_topo,water_level,ponds,pc_drain,time_elapsed_draining = drain(holes = holes,\
    ice_topo = ice_topo,ponds = ponds,water_level = water_level,sea_level = -np.inf,DH = 0.1,DH_drain = DH_drain, H = 1., dt = dt_drain,\
    Tdrain = Tdrain,Tmelt = np.inf, conn = 8, separate_timescales = True,hydrostatic_adjustment = False, eps = 1E-10) 

# find disconnected ponds  
ret, L = cv2.connectedComponents(ponds,connectivity = 8)


cdict1 = {'red':   ((0.0, 0.2, 0.2),
                   (0.5, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.2, 0.2),
                   (0.5, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

         'blue':  ((0.0, 1.0, 1.0),
                   (0.5, 1.0, 1.0),
                   (1.0, 1.0, 1.0))
        }

pond_cmap = LinearSegmentedColormap('PondCmap', cdict1)

# plot topography, ponds after drainage, and disconnected ponds
plt.figure(1)
plt.clf()
plt.subplot(1,3,1)

plt.imshow(ice_topo)

plt.subplot(1,3,2)
plt.imshow(1-ponds,cmap = pond_cmap)
plt.scatter(holes[:,0],holes[:,1],c='r')

plt.subplot(1,3,3)
plt.imshow(L)
plt.scatter(holes[:,0],holes[:,1],c='r')

plt.show()