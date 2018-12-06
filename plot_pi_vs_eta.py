import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from hole_function_defs import *

"""
This is the code to generate Fig. 2 of the paper "Critical percolation threshold is an upper bound on Arctic sea ice melt pond coverage." 
To run this code, 2d_hole_model must be run previously and a HoleDrain dictionary should be saved. 
"""

filename = 'Path/to/HoleDrain'  # Path to HoleDrain dictionary
version = 0                     #Version of HoleDrain to load


pthresh = {'diffusion': 0.5, 'rayleigh': 0.4, 'snow_dune': 0.465}    # percolation thresholds for different topographies
c = {'diffusion': 1.2, 'rayleigh': 1., 'snow_dune': 1.}              # constants c for different topographies

col = ['r','b','g','y','c','m','k']
plt.figure(1)
plt.clf()

HoleDrain = load_Dict(filename,version)
record_mean_evolution = HoleDrain['metadata']['record_mean_evolution']
if record_mean_evolution:     
    
    leg = []
    # Loop through all of the parameter combinations in this dictionary
    for j in range(len(HoleDrain['Record_list'])):
        
        # Extract parameters of each run
        res = HoleDrain['Record_list'][j]['res']
        mode = HoleDrain['Record_list'][j]['mode']
        l0_c = np.mean(HoleDrain['l0'][j])
        N0 = res**2
        
        p = np.array(HoleDrain['pc_mean_evolution'][j])
        N = np.array(HoleDrain['Nholes_mean_evolution'][j])
        
        pi = p/pthresh[mode]
        eta = N*l0_c**2/res**2*c[mode]
        
        plt.subplot(1,2,1)
        plt.semilogx(N,p,col[j] + '-',lw = 2)
        
        plt.xlabel('Number of holes',fontsize = 18)
        plt.ylabel('Pond coverage',fontsize = 18)
    
        
        plt.subplot(1,2,2)
        plt.semilogx(eta,pi,col[j] + '-',lw = 2)
        
        leg.append(mode)
        plt.xlabel('eta',fontsize = 18)
        plt.ylabel('pi',fontsize = 18)
    
    plt.legend(leg,frameon = False)

plt.show()
    