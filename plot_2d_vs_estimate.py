import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from hole_function_defs import *

"""
This is the code to generate Fig. 3 of the paper "Critical percolation threshold is an upper bound on Arctic sea ice melt pond coverage." 
To run this code, 2d_hole_model must be run previously and a HoleDrain dictionary should be saved. 
"""

# Load a HoleDrain dictionary to estimate g(eta). If version_gx < 0, Eq. 4 of the paper is used to estimate g(eta)
filename_gx = 'Path/to/HoleDrain' # Path to HoleDrain dictionary used to evaluate g(eta)
version_gx = 0      # Version of HoleDrain to load to evaluate g(eta)
idx = 0             # choose combination of parameters in the 2d hole model run to estimate g(eta)

filename_pd = 'Path/to/HoleDrain' # Path to HoleDrain dictionary used to compare post-drainage pond coverage in 2d model to estimated post-drainage pond coverage (Fig. 3a)
version_pd = 0      # Version of HoleDrain to load to compare post-drainage pond coverage in 2d model to estimated post-drainage pond coverage (Fig. 3a)

filename_evol = 'Path/to/HoleDrain' # Path to HoleDrain dictionary used to compare pond evolution in the 2d model to estimated pond evolution (Fig. 3b)
version_evol = 0    # Version of HoleDrain to load to compare pond evolution in the 2d model to estimated pond evolution (Fig. 3b)
idx_evol = 0       # choose combination of parameters to compare pond evolution in the 2d model to estimates

pthresh = {'diffusion': 0.5, 'rayleigh': 0.4, 'snow_dune': 0.465}    # percolation thresholds for different topographies
c = {'diffusion': 1., 'rayleigh': 1.2, 'snow_dune': 1.}              # constants c for different topographies

# Get universal function g(eta), by interpolating a single run of the 2d hole model. To get a reliable interpolation,
# the 2d model should be run with no ponded ice melt (Tmelt = np.inf) and no ice thinning (include_ice_thinning = False)
x,gx = get_gx(filename = filename_gx,version = version_gx, idx = idx, xmin = 0.0003, pthresh = pthresh, c = c) 

plt.figure(1)       # plot g(eta)
plt.clf()
plt.semilogx(x,gx(x),'r-',lw=2)
    
    
p_estimated_list = []
p_simulated_list = []
N_estimated_list = []
Nx_estimated_list = []
N_simulated_list = []
Nx_simulated_list = []
TmTh_list = []

"""  
Load HoleDrain to compare estimated and the 2d model post-drainage pond coverage (Fig. 3a of the paper). 
Should contain various combinations of parameters but no ice thinning (include_ice_thinning = False)
"""
HoleDrain = load_Dict(filename_pd,version_pd)
record_mean_evolution = HoleDrain['metadata']['record_mean_evolution']  # Check if the needed variables were recorded in HoleDrain
if record_mean_evolution:     
    
    # Loop through all of the parameter combinations in this dictionary
    for j in range(len(HoleDrain['Record_list'])):
        
        # Extract parameters of each run
        res = HoleDrain['Record_list'][j]['res']
        mode = HoleDrain['Record_list'][j]['mode']
        Tmelt = HoleDrain['Record_list'][j]['Tmelt']
        Thole = HoleDrain['Record_list'][j]['Thole']
        l0_c = np.mean(HoleDrain['l0'][j])
        N0 = res**2
                
        p = pthresh[mode] * find_ppc(gx,N0,l0_c,res,Tmelt,Thole,pthresh[mode],eps = 1E-10)   # Make an estimate. Simultaneously solve p = p_c * g(eta) and Tm = Tmelt/(1-p)  
        
        p_estimated_list.append(p)
        p_simulated_list.append(np.mean(HoleDrain['pc_after_drainage'][j]))    
    
    plt.figure(2)           # plot pd_estimated vs pd_simulated
    plt.clf()
    xx = np.linspace(0,0.5,100)
    plt.plot(np.array(p_estimated_list),np.array(p_simulated_list),'bs')
    plt.plot(xx,xx,'r--',lw = 2)
    
    plt.xlabel('Estimated pond coverage',fontsize = 18)
    plt.ylabel('Simulated pond coverage',fontsize = 18)

# Load HoleDrain to compare estimated and the 2d model pond evolution (Fig. 3b of the paper). 
HoleDrain = load_Dict(filename_evol,version_evol)
record_mean_evolution = HoleDrain['metadata']['record_mean_evolution']  # Check if the needed variables were recorded in HoleDrain
if record_mean_evolution:     
    
    # Parameters used in simulations
    res = HoleDrain['Record_list'][idx_evol]['res']
    mode = HoleDrain['Record_list'][idx_evol]['mode']
    Tmelt = HoleDrain['Record_list'][idx_evol]['Tmelt']
    Thole = HoleDrain['Record_list'][idx_evol]['Thole']
    l0_c = np.mean(HoleDrain['l0'][idx_evol])
    dHdt = HoleDrain['Record_list'][idx_evol]['dHdt']/3600.
    H = HoleDrain['Record_list'][idx_evol]['H']
    N0 = res**2
          
    day = 24.*3600.
    
    # Simulated pond evolution
    p_simulated = np.array(HoleDrain['pc_mean_evolution'][idx_evol])
    N_simulated = np.array(HoleDrain['Nholes_mean_evolution'][idx_evol])
    t_simulated = HoleDrain['metadata']['time_eval']
    
    """
    In order to estimate t0 accurately, the 2d simulation should be run until
    all holes open. If this was not done, t0 can be estimated using Eq. 9 of the paper
    by setting estimate_t0 = True
    """
    # uncomment for an alternative way to calculate t0 in simulations
    """
    loc_mean_hole_distribution = np.argmax(np.gradient(N_simulated))
    t0 = t_simulated[loc_mean_hole_distribution]*3600
    """
    t0 = np.sum(t_simulated*np.gradient(N_simulated))/N0 *3600
    
    # Estimate pond evolution. All of the parameters must be in units of m and s.
    time_estimate,p_estimate, _ = get_pond_evolution_stage_II_III(filename = filename_gx, Tm = Tmelt*3600., Th = Thole*3600., Tdrain = 0.*day, \
                                                                t0 = t0, N0 = N0,  pthersh = pthresh[mode], c = c[mode], l0 = l0_c, L = res, H = H, dHdt = dHdt, version = version_gx, idx = idx, \
                                                                estimate_t0 = False, use_physical_params = False, physical_params = {}, \
                                                                tsteps = 100, t_initial = 0.*day, tmax = 100.*day, pdf = 'normal', param = [1.], pinit = pthresh[mode])
    
    plt.figure(3) # Plot pond evolution      
    plt.clf()
    
    plt.plot(np.array(t_simulated)/24.,np.array(p_simulated),'b-',lw = 3)
    plt.plot(time_estimate,p_estimate,'r--',lw = 3)
    
    plt.xlabel('Time [days]',fontsize = 18)
    plt.ylabel('Pond coverage',fontsize = 18)
    plt.legend(['Simulated pond evolution','Estimated pond evolution'],frameon = False)
    plt.ylim(0,0.8)
    plt.xlim(0,50)


plt.show()
    