import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import interp1d
import itertools
import re
import scipy.stats as stats
from hole_function_defs import *

"""
 This code runs the 2d model of Arctic sea ice melt pond drainage through holes that open randomly on a synthetic topography. The code was tested on a 64-bit Mac 
 using python 2.7.10. The model is described in detail in the paper "Critical percolation threshold is an upper bound on Arctic sea ice melt pond coverage"
 
 Library floodfill (https://github.com/cgmorton/flood-fill) must be installed to run this code. Additionally, AstroML (https://github.com/astroML/astroML) should 
 also be installed to calculate the pond autocorrelation function. If it is not installed, set record_corr = False

 Output of this code is a dictionary, HoleDrain, that contains all of the defining parameters of the model as well as variables summarizing the resulting pond evolution.
 The code initializes a composite run that consits of several individual runs with different parameter combinations.The composite run is stored in the HoleDrain dictionary.
 Parameters of the composite run are given in HoleDrain['metadata']. To change these variables, change entries in HoleDrain['metadata'] directly.
 Some of the parameters in HoleDrain['metadata'] are lists (e.g. res_list, sigma_h_list,...) and these lists define different parameter combinations of individual runs. 
 The code loops through all of the combinations of parameters given in these lists. For example, if res_list = [500,1000], sigma_h_list = [0.01,0.03], and mode_list = ['snow_dune'],
 a composite run will consist of 4 individual runs with parameter (res = 500, sigma_h = 0.01, mode = 'snow_dune'), (res = 1000, sigma_h = 0.01, mode = 'snow_dune'), 
 (res = 500, sigma_h = 0.03, mode = 'snow_dune'), and (res = 1000, sigma_h = 0.03, mode = 'snow_dune'). All of the individual run parameters are stored in 
 dictionaries HoleDrain['Record_list'][i]. For example, parameter res for ith individual run can be retrieved as HoleDrain['Record_list'][i]['res']. Optionally,
 each individual run may be repeated several times for the same combination of parameters by changing the variable realizations. The only difference between these different
 realizations are then stochastic elements such as the random realization of the topography or the locations of holes. 
 
 
 Variables summarizing the resulting pond evolution to be stored in HoleDrain are defined in the Vars list. These variables can be accessed after the 
 code finishes. Pond evolution variables during i-th individual run and jth realization can be accessed as var = HoleDrain['var'][i][j]. 
 For example, accessing ice topography of the jth realization of the ith individual run can be done as HoleDrain['ice_topo_record'][i][j]. Some variables are 
 recorded as means over different realizations and are only stored for individual runs. For example accessing the mean pond coverage of ith realization 
 can be done as HoleDrain['pc_mean_evolution'][i].  

 If save = True, HoleDrain will be saved as .pickle in a directory and by name defined in filename. Even if filename is the same for two different composite runs, 
 the code will not overwrite previously saved dictionaries but will instead create a new version (e.g. if HoleDrain_v1 exists, code will create HoleDrain_v2). 
 Each of these versions can be loaded by changing the variable version. 

To change the variables of the composite run, entries in HoleDrain['metadata'] should be changed directly. Each variable that contains _list will be looped over.
Meanings of the parameters are the following

Characteristics of the topography:
    
    res_list - domain size in pixels. Model is run on a grid of size (res,res)
    hw - pond initializion. Currently only option is 'max', which sets the initial pond water level above maximum ice height
    sea_level - sea level height. Must be set to 0 to work properly
    H - initial ice thickness in m
    sigma_h_list - standard deviations of ice topography in m. Currently seems to be a problem with large sigma_h 
    mode_list - topography types. Can choose between 'snow_dune', 'diffusion', and 'rayleigh'
    tmax - If the topography type is 'diffusion' or 'rayleigh' for how long to diffuse the surface. This defines the length-scale for 'diffusion' and 'rayleigh' topographies
    g - anisotropy factor. 1 stands for isotropic topography
    conn_list - defines neighborhood when determining connectedness. Can be 4 (4-neighborhood) or 8 (8-neighborhood)
    snow_dune_radius_list - mean radius of mounds in pixels if mode = 'snow_dune'
    Gaussians_per_pixel_list - mound density if mode = 'snow_dune'. Defined as (Number of mounds) * snow_dune_radius**2 / res**2
    number_of_r_bins - how many radii categories to consider when approximating the snow dune topography if mode = 'snow_dune'
    window_size - cutoff parameter when approximating the snow dune topography if mode = 'snow_dune'
    snow_dune_height_exponent - exponent that relates height of mounds to their radius if mode = 'snow_dune' (height is proportional to radius^snow_dune_height_exponent). Default is 1
    
Physical constants:
    
    rho_w - density of water in kg/m^3
    rho_i - density of ice in kg/m^3
    Latent_heat - latent heat of melting in J/kg
    
Parameters of the model run:
    
    hole_opening_method_list - schemes for opening holes. Can be 'hole_potential' (each pixel is a potential hole and there can only be one hole per pixel) or 'random_process' (multiple holes per pixel allowed)
    hole_fun_list - hole opening distributions, F. Can be 'erf' (error function), 'exp' (exponential), or 'linear' if hole_opening_method = 'hole_potential', or 'erf' or 'exp' if hole_opening_method = 'random_process'.
    Tdrain - Time, in hours, to drain a pond through a hole.
    Tmelt_list - time, in hours, for ponded ice to melt through (rho_w-rho_i)/rho_w * H. Defines the rate of ponded ice melt, dh_diff/dt 
    Thole_list - hole opening timescale in hours
    separate_timescales - if True, draining happens separately from melting (equivalent to Tdrain << Tmelt). Case when separate_timescales = False is not reliably tested
    hydrostatic_adjustment - whether to maintain ice in hydrostatic balance
    realizations - each run with a given set of parameters may be repeated a number of times. Number of times to repeat each run
    include_ice_thinning - whether to decrease ice thickness over the course of the run
    dHdt_list - ice thinning rate in m/hour.  
                    
Cutoffs:    
    
    hole_number - possible number of holes. If hole_opening_method = 'hole_potential', this is automatically changed to res**2
    min_hole_number - minimum number of holes that have to open before the current run can be stopped
    pc_cutoff - pond coverage below which the current run is stopped
    pond_bottom_cutoff - stop the current run if the maximum height of pond bottoms falls below pond_bottom_cutoff
    dt_drain - time step, in hours, for performing drainage
    dt_step - time step for opening new holes
    dt_topography - if topography is 'diffusion' or 'rayleigh' time step for diffusion
    eps - initialize pond water table at max(topography) + eps
    eps_drain - cutoff to identify ponds at sea level
    eps_hydrostatic - cutoff for maintaining hydrostatic balance after ice thinning
    t_cutoff - time, in hours, after which the current run is stopped. If t_cutoff < 0 this stopping condition is inactive. If t_cutoff > 0, this is the only stopping condition and others become inactive. Should be set > 0 if include_ice_thinning = True 

Recording parameters:  
    
    record_every_step - if True record pond coverage, number of holes, hydraulic head, and the location of all the holes at each time step. Takes a lot of memory, default is False
    record_mean_evolution - if True, record mean and standard deviation between all realizations of pond coverage, number of holes, and hydraulic head evaluated at time_eval. Uses much less memory than record_every_step without significant loss of information
    record_topography - if True, save each initial topography
    time_eval - if record_mean_evolution = True, evaluate pond coverage, number of holes, and hydraulic head at these times
    record_corr - whether to record the autocorrelation function of ponds
    corr_bins - bins at which to evaluate the autorrelation function
    random_fraction - fraction of pixels to be included in claculating autocorrelation function. If the number of pixels is too high calculating autocorrelation may take a significant amount of time
    hole_count_for_recording_AP - number of holes that should open to record geometric characteristics of ponds (area and perimeter). Recording is done only once per realization.
    

Variables that summarize the resulting pond evolution (denoted as strings in Vars) are recorded in HoleDrain after the end of the composite run. These variables are
    
    Record_list - list of dictionaries containing parameters for each individual run
    holes_history - locations of holes at each time step. Only recorded if record_every_step = True
    hh - mean hydraulic head across all ponds at each time step. Only recorded if record_every_step = True
    pc - pond coverage at each time step. Only recorded if record_every_step = True
    Nholes - number of holes at each time step. Only recorded if record_every_step = True
    time - current time at each time step. Only recorded if record_every_step = True
    Nholes_mean_evolution - mean (among different realizations with the same parameters) of the number of holes evaluated at t_eval. Only recorded if record_mean_evolution = True
    Nholes_std_evolution - standard deviation (among different realizations with the same parameters) of the number of holes evaluated at t_eval. Only recorded if record_mean_evolution = True
    hh_mean_evolution - mean (among different realizations with the same parameters) of the hydraulic head evaluated at t_eval. Only recorded if record_mean_evolution = True
    hh_std_evolution - standard deviation (among different realizations with the same parameters) of the hydraulic head evaluated at t_eval. Only recorded if record_mean_evolution = True
    pc_mean_evolution - mean (among different realizations with the same parameters) pond coverage evaluated at t_eval. Only recorded if record_mean_evolution = True
    pc_std_evolution - standard deviation (among different realizations with the same parameters) pond coverage evaluated at t_eval. Only recorded if record_mean_evolution = True
    pc_after_drainage - pond coverage at the end of each realization
    hh_after_drainage - hydraulic head at the end of each realization
    number_of_holes - number of holes opened at the end of each realization
    time_elapsed - time at the end of end of each realization
    corr - autocorrelation function for each individual run. Only recorded if record_corr = True. Recorded for every realization
    d_corr - distances at which autocorrelation function is estimated. Recorded for every realization
    l0 - autocorrelation length (distance at which corr = 1/e). Recorded for every realization
    ice_topo_record - initial ice topography for each individual run. Only evaluated if record_topography = True. Recorded for every realization
    As_initial_list - pond areas after drainage through hole_count_for_recording_AP holes. Recorded once per realization
    Ps_initial_list - pond perimeters after drainage through hole_count_for_recording_AP holes. Recorded once per realization
    
"""


"""
 Get variables from dictionary keys. Dict_name should be given as a string (e.g. 'HoleDrain')
 For example:
     Dict = {'a':5, 'b':15}
     get_variables('Dict')
 gives:
     a = Dict['a']
     b = Dict['b']
"""
def get_variables(Dict_name):
    code = 'keys = ' + Dict_name + '.keys()'
    exec(code,globals())
    for key in keys:
        code = key + ' = ' + Dict_name + '["%s"]'%key
        exec(code, globals()) 
"""
 Collect variables into a dictionary. Vars should be given as a list of strings of names of variable to be collected into a dictionary also given as a string.
 For example:
     a = 5; b = 15; Dict = dict()
     Vars = ['a','b']; 
     collect_variables(Vars,'Dict')
 gives:
     Dict['a'] = a
     Dict['b'] = b   
"""
def collect_variables(Vars,Dict_name):
    for Var in Vars:
        code = Dict_name+'["%s"]'%Var + ' = ' + Var
        exec(code)

# Substitute a substring ~substring_old~ in each string in ~strings~ with ~substring_new~
def substitute_substring(strings,substring_old,substring_new):
    new_strings = []
    for string in strings:
        new_strings.append(re.sub(substring_old,substring_new,string))
    return new_strings
  

#  ----------  Initialize all variables -----------
res_list = 0; hw = 0.; sea_level = 0; hole_number = 0; mode_list = 0; tmax = 0; dt = 0; dt_topography = 0.; g = 0;
Tdrain = 0.; Tmelt_list = 0.; Thole_list = 0.; separate_timescales = 0; realizations = 0;
record_every_step = 0; hydrostatic_adjustment = 0; H = 0; rho_w = 0;
rho_i = 0; sigma_h = 0; eps = 0; min_hole_number = 0; dt_drain = 0; dt_step = 0;
hole_opening_method = 0; hole_fun = 0; pc_cutoff = 0; pond_bottom_cutoff = 0;
corr_bins = 0; random_fraction = 0; record_corr = 0; record_mean_evolution = 0; time_eval = 0;
eps_drain = 0; record_topography = 0; conn_list = 0; dHdt_list = 0; include_ice_thinning = 0;
t_cutoff = 0; hole_opening_method_list = 0; hole_fun_list = 0; snow_dune_radius = 0; 
Gaussians_per_pixel_list = 0; number_of_r_bins = 0; window_size = 0; snow_dune_height_exponent = 0;
hole_count_for_recording_AP = 0; snow_dune_radius_list = 0; sigma_h_list = 0; eps_hydrostatic = 0;
# ---------------------------------------------------

run = True      # Run the code? Should not run if load = True
save = True     # Save HoleDrain dictionary?
load = False    # Load a previously saved HoleDrain dictionary?

# path and filename for saving and loading HoleDrain
filename = 'Path/To/File/HoleDrain'

# Which previously saved version of HoleDrain to load. If just one exists, version should be 0
version = 0

if load:
    HoleDrain = load_Dict(filename,version)
    get_variables("HoleDrain")
    get_variables("HoleDrain['metadata']")
    
else:    
    
    HoleDrain = dict()
    
    # Define characteristics of the topography:
    HoleDrain['metadata'] = {'res_list': [500], 'hw': 'max','sea_level': 0.,'H': 1.5, 'sigma_h_list': [0.03], 
                            'mode_list': ['snow_dune'], 'tmax': 2., 'g': 1., 'conn_list': [8],
                            'snow_dune_radius_list': [1.], 'Gaussians_per_pixel_list': [0.2], 'number_of_r_bins': 150, 
                            'window_size': 5, 'snow_dune_height_exponent': 1.}
    
    # Define physical constants:
    HoleDrain['metadata'].update({  'rho_w':1000.,'rho_i':900., 'Latent_heat': 3.006E8 })
    
    # Define parameters of the model run:
    HoleDrain['metadata'].update({ 'hole_opening_method_list': ['hole_potential'], 'hole_fun_list': ['erf'], 
                                   'Tdrain': 10., 'Tmelt_list': [360.,np.inf], 'Thole_list': [120.],'separate_timescales': True,
                                   'hydrostatic_adjustment': True, 'realizations': 1,'include_ice_thinning': False, 'dHdt_list': np.array([0.]) /100./24.})
    
    # Define cutoffs:                      
    HoleDrain['metadata'].update({'hole_number': 500**2, 'min_hole_number': 5, 'pc_cutoff': 0.01, 'pond_bottom_cutoff': -0.02, 'dt_drain': 0.1, 
                                'dt_step': 1.,'dt_topography': 0.1, 'eps': 1.E-5, 'eps_drain': 1E-10, 'eps_hydrostatic': 1E-10, 't_cutoff':-1})
    
    # Define recording parameters:
    HoleDrain['metadata'].update({'record_every_step': False, 'record_mean_evolution': True, 'record_topography': True, 'time_eval': np.linspace(0.1,1500.,1000),
                                  'record_corr': True, 'corr_bins': np.linspace(0,40,100), 'random_fraction': 0.5, 'hole_count_for_recording_AP': 10})
    
    # Comments                                                    
    HoleDrain['comments'] = []
   
    # Remove the substring _list from parameters in HoleDrain['metadata'] to record them in each individual run        
    Record_Vars = substitute_substring(HoleDrain['metadata'].keys(),'_list','')
    
    # Convert parameters in HoleDrain['metadata'] to Python variables
    get_variables("HoleDrain['metadata']")
    
    # Variables to be saved in HoleDrain after the code finishes
    Vars = ['holes_history', 'hh', 'pc','Nholes_mean_evolution','hh_mean_evolution','pc_mean_evolution', \
            'Nholes_std_evolution', 'hh_std_evolution', 'pc_std_evolution',\
            'Record_list', 'time', 'Nholes', 'time_elapsed',\
            'pc_after_drainage', 'hh_after_drainage', 'number_of_holes',\
            'corr','d_corr','l0','ice_topo_record','As_initial_list','Ps_initial_list']

if run:
    
    # initialize all variables to be recorded
    holes_history = []; Nholes = []; time = []; hh = []; pc = [];
    Nholes_mean_evolution = []; hh_mean_evolution = []; pc_mean_evolution = [];
    Nholes_std_evolution = []; hh_std_evolution = []; pc_std_evolution = [];
    As_initial_list = []; Ps_initial_list = [];
        
    Record_list = []
    
    ice_topo_record = []
    
    pc_after_drainage = []
    hh_after_drainage = []
    number_of_holes = []
    time_elapsed = []
    
    corr = []
    d_corr = []
    l0 = []
    
    index = 0   # counter for current parameter combination
    
    # loop through all combinations of parameters in lists defined in HoleDrain['metadata']
    for imelt, ihole, imode, i_res,i_conn,i_dHdt,i_hole_opening_method,i_hole_fun, i_Gaussians_per_pixel, i_snow_dune_radius, i_sigma_h in \
        itertools.product(range(len(Tmelt_list)),range(len(Thole_list)),range(len(mode_list)),range(len(res_list)),range(len(conn_list)),\
        range(len(dHdt_list)),range(len(hole_opening_method_list)),range(len(hole_fun_list)),range(len(Gaussians_per_pixel_list)),\
        range(len(snow_dune_radius_list)),range(len(sigma_h_list))):
        
        # Individual run parameters
        Tmelt = Tmelt_list[imelt]
        Thole = Thole_list[ihole]
        mode = mode_list[imode]
        res = res_list[i_res]
        conn = conn_list[i_conn]
        dHdt = dHdt_list[i_dHdt]
        hole_opening_method = hole_opening_method_list[i_hole_opening_method]
        hole_fun = hole_fun_list[i_hole_fun]
        Gaussians_per_pixel = Gaussians_per_pixel_list[i_Gaussians_per_pixel]
        snow_dune_radius = snow_dune_radius_list[i_snow_dune_radius]
        sigma_h = sigma_h_list[i_sigma_h]
        
        # if hole_opening_method == 'hole_potential' number of potential holes is the number of pixels
        if hole_opening_method == 'hole_potential':
            hole_number = res**2
        else:
            hole_number = HoleDrain['metadata']['hole_number']
        
        # define thickness DH through which ponded ice melts after Tmelt
        DH = H * (rho_w - rho_i)/rho_w 
        
        # all of the parameters of the individual runs will be stored in Record_dict 
        Record_dict = dict()
        # collect all of the parameters of the current run in Record_dict 
        collect_variables(Record_Vars,'Record_dict')
        # append Record_dict to a list that will be recorded in HoleDrain at the end of the composite run
        Record_list.append(Record_dict)
        
        # for each combination of parameters, initialize variables to be recorded after each realization
        ice_topo_record.append([])
        
        pc_after_drainage.append(np.zeros(realizations))
        hh_after_drainage.append(np.zeros(realizations))
        number_of_holes.append(np.zeros(realizations))
        time_elapsed.append(np.zeros(realizations))
        holes_history.append([]); Nholes.append([]); time.append([]); hh.append([]); pc.append([]); 
        
        As_initial_list.append(np.array([])); Ps_initial_list.append(np.array([]));
        corr.append(np.zeros([realizations,len(corr_bins)+1]))
        d_corr.append(np.zeros([realizations,len(corr_bins)+1]))
        l0.append(np.zeros(realizations))
        
        Nholes_mean_evolution.append(np.zeros(len(time_eval))) 
        hh_mean_evolution.append(np.zeros(len(time_eval)))
        pc_mean_evolution.append(np.zeros(len(time_eval)))
        Nholes_std_evolution.append(np.zeros(len(time_eval))) 
        hh_std_evolution.append(np.zeros(len(time_eval))) 
        pc_std_evolution.append(np.zeros(len(time_eval))) 
        
        if record_mean_evolution:
            
            Nholes_mean_evolution_realiztions = np.zeros([realizations,len(time_eval)])
            hh_mean_evolution_realiztions = np.zeros([realizations,len(time_eval)])
            pc_mean_evolution_realiztions = np.zeros([realizations,len(time_eval)])
        
        # for each combination of parameters loop through realizations
        for j in range(realizations):
            
            # create initial topography
            ice_topo = Create_Initial_Topography(res = res,mode = mode,tmax = tmax,dt = dt_topography,g = g,sigma_h = sigma_h,h = 0.,snow_dune_radius = snow_dune_radius, Gaussians_per_pixel = Gaussians_per_pixel, 
                              number_of_r_bins = number_of_r_bins, window_size = window_size, snow_dune_height_exponent = snow_dune_height_exponent)

            # if record_corr = True, cut initial topography with a plane so that pond coverage = 0.5 and find the autocorrelation of the resulting pond configuration.
            if record_corr:
                
                # find ponds with pond coverage = 0.5
                m = np.max(ice_topo)
                dm = 0.01*(np.max(ice_topo) - np.min(ice_topo))
                ponds = np.array([1])
                while np.mean(ponds) > 0.5:
                    m -= dm
                    ponds = ice_topo < m
                
                # find autocorrelation
                Correlation_fun,Correlation_dist,Correlation_len = find_corr(ponds = ponds,random_fraction = random_fraction,bins = corr_bins)
                
                # record autocorrelation
                corr[-1][j,:] = Correlation_fun
                d_corr[-1][j,:] = Correlation_dist
                l0[-1][j] = Correlation_len
                print('Corr length = ' + str(Correlation_len))
            
            # set initial pond coverage at 1 by setting the water level at np.max(ice_topo) + eps, shift the topography to maintain hydrostatic balance, and initialize ponds
            if hw == 'max':
                w0 = (rho_w-rho_i)/rho_w*H
                h = w0 - ( np.max(ice_topo) + eps )
                ice_topo += h
                
                ponds = (ice_topo < w0).astype(np.uint8)
            
            # Initialize the water level field
            water_level = w0 * np.ones([res,res])
            ice = (1-ponds).astype(bool)
            water_level[ice] = ice_topo[ice]
            
            # Initialize variables to be recorded after each time step
            holes_history[-1].append([]); Nholes[-1].append([0.]); time[-1].append([0.]);
            hh[-1].append([w0-sea_level]); pc[-1].append([np.mean(ponds)]);
            
            # if hole_opening_method = 'hole_potential', for each pixel define a threshold hole_potential after which it becomes a hole
            if hole_opening_method == 'hole_potential':
                if hole_fun == 'erf':
                    hole_potential = np.random.normal(size = (res,res))
                if hole_fun == 'linear':
                    hole_potential = np.random.rand(res,res)
                if hole_fun == 'exp':
                    hole_potential = -np.random.exponential(size = (res,res))
                
                # initial hole_potential is below the threshold of all pixels
                h_potential = np.min(hole_potential)
            
            # if hole_opening_method = 'random_process', define a hole opening probability density function that defines the number of holes that open at each time step
            if hole_opening_method == 'random_process':
                if hole_fun == 'erf':
                    h_potential = stats.norm.ppf(0.05/hole_number)
                if hole_fun == 'exp':
                    h_potential = -stats.expon.ppf(1.-0.05/hole_number)
            
            
            hole_count = 0      # current number of holes
            time_current = 0    # current time
            count = 0           # current step
            cond = True         # when cond becomes False, stop current run
            
            if record_mean_evolution:
                
                time_history = []
                Nholes_mean_evolution_history = []
                hh_mean_evolution_history = []
                pc_mean_evolution_history = []  
            
                         
            H_current = H       # current ice thickness
            holes = []          # coordinates of all holes
            time_elapsed_draining = 0
            flag = True
            
            # run the drianage model until cond = Falses
            while cond:
                
                time_current += dt_step
                
                # open new holes at random locations according to hole_opening_method
                if hole_opening_method == 'hole_potential':
                    h_potential += dt_step/Thole
                    holes = np.array(np.nonzero(hole_potential < h_potential)).T
                    
                    hole_count = len(holes)
                
                if hole_opening_method == 'random_process':
                    h_potential += dt_step/Thole
                    if hole_fun == 'erf':
                        lambda_new = hole_number*stats.norm.pdf(h_potential)*dt_step/Thole
                    if hole_fun == 'exp':
                        lambda_new = hole_number*stats.expon.pdf(-h_potential)*dt_step/Thole
                    N_new = stats.poisson.rvs(mu = lambda_new, size = 1)[0]
                    
                    if N_new > 0:
                        
                        new_holes = np.random.randint(0,res,size = (N_new, 2))
                        if len(holes) > 0:
                            holes = np.vstack([holes,new_holes])
                            holes = np.vstack({tuple(row) for row in holes})
                        else:
                            holes = new_holes
                    
                    hole_count += N_new
                
                # print(current coordinate combination, current realization, number of open holes, maximum ice height of ponded ice, pond coverage)
                print(index,j,hole_count,np.max(ice_topo[ponds.astype(bool)]),np.mean(ponds))
                
                # if there are open holes, perform drainage
                if len(holes) > 0:
                    
                    # define pond depth to be drained during Tdrain
                    DH_drain = np.max(np.min([water_level[holes[:,0],holes[:,1]] - sea_level, water_level[holes[:,0],holes[:,1]] - ice_topo[holes[:,0],holes[:,1]]],axis = 0))
                    
                    # drain all ponds connected to holes
                    ice_topo,water_level,ponds,pc_drain,time_elapsed_draining = drain(holes = holes,\
                        ice_topo = ice_topo,ponds = ponds,water_level = water_level,sea_level = sea_level,DH = DH,DH_drain = DH_drain, H = H_current, dt = dt_drain,\
                        Tdrain = Tdrain,Tmelt = Tmelt,conn = conn,separate_timescales = separate_timescales,hydrostatic_adjustment = hydrostatic_adjustment, eps = eps_drain)    
               
                # preferentially melt ponded ice
                ice_topo = melt(ice_topo,ponds,dt_step,Tmelt,DH)    
                
                # if timescales of draining and melting are not separated, we have to take into account the time spent draining
                if not separate_timescales:
                     time_current += time_elapsed_draining
                
                # if include_ice_thinning, change current ice thickness by -dHdt * dt_thin 
                # and then restore hydrostatic balance while keeping ponds drained  
                if include_ice_thinning:
                    if separate_timescales:
                        dt_thin = dt_step
                    else:
                        dt_thin = dt_step+time_elapsed_draining
                    
                    H_current -= dHdt * dt_thin 
                    
                    ice_topo_h = ice_topo.copy()
                    water_level_h = water_level.copy()
                    
                    dw1 = -H_current
                    dw2 = H_current
                    dw = 0.5*(dw1+dw2)
                    
                    ice_topo_h = ice_topo + dw
                    water_level_h = water_level + dw
                    
                    ponds_sea_level_cut = ice_topo_h <= sea_level
                    ret, L = cv2.connectedComponents(ponds_sea_level_cut.astype(np.uint8),connectivity = conn) 
                    
                    ponds_sea_level_cut_labels = np.array(list(set(np.unique(L[holes[:,0],holes[:,1]])) - {0}))
                    ponds_sea_level_cut_mask = (np.in1d(L,ponds_sea_level_cut_labels).reshape(np.shape(L)[0],np.shape(L)[1])).astype(bool)                    
                    
                    water_level_h[ponds_sea_level_cut_mask] = sea_level
                    wl = np.mean(water_level_h)
                    while np.abs((rho_w - rho_i)/rho_w * H_current - wl) > eps_hydrostatic:
            
                        if wl > (rho_w - rho_i)/rho_w * H_current:
                            dw2 = dw
                        else:
                            dw1 = dw
                        dw = 0.5*(dw1+dw2)
                        
                        ice_topo_h = ice_topo + dw
                        water_level_h = water_level + dw
                        
                        ponds_sea_level_cut = ice_topo <= sea_level
                        ret, L = cv2.connectedComponents(ponds_sea_level_cut.astype(np.uint8),connectivity = conn) 
                        
                        ponds_sea_level_cut_labels = np.array(list(set(np.unique(L[holes[:,0],holes[:,1]])) - {0}))
                        ponds_sea_level_cut_mask = (np.in1d(L,ponds_sea_level_cut_labels).reshape(np.shape(L)[0],np.shape(L)[1])).astype(bool)                    
                        
                        water_level_h[ponds_sea_level_cut_mask] = sea_level
                        wl = np.mean(water_level_h)                 
                    
                    ice_topo = ice_topo_h.copy()
                    water_level = water_level_h.copy()    
                    ponds = (ice_topo < water_level).astype(np.uint8)
                    
                    
                # calculate mean hydraulic head
                hydraulic_head = np.mean(water_level[ponds.astype(bool)]) - sea_level
                
                # find pond area and perimeter after hole_count_for_recording_AP holes open. Do this only once (flag becomes False)
                if flag & (hole_count > hole_count_for_recording_AP):
                    flag = False
                    As, Ps = FindAreaPerimeter(ponds.astype(np.uint8),mpp=1.,mask = [])
                    As_initial_list[-1] = np.hstack([As_initial_list[-1] , As])
                    Ps_initial_list[-1] = np.hstack([Ps_initial_list[-1] , Ps])
                
                # check the stopping condition
                if np.sum(ponds) > 0:
                    
                    # if t_cutoff < 0, stopping condition is: (open at least minimum number of holes) 
                    # and either one of ( (open too many holes), (ponds irreversably melt below sea level), 
                    # (mean hydraulic head drops below 10% of initial freeboard), or 
                    # (pond coverage gets too low), or (ice gets too thin) )
                    cond = (hole_count < min_hole_number) or ( (hole_count < hole_number) and ( np.max(ice_topo[ponds.astype(bool)]) > pond_bottom_cutoff ) \
                            and ( hydraulic_head > 0.1*w0 ) and (np.mean(ponds) > pc_cutoff)) and (H_current > 30*dHdt*dt_step)
                    print(hole_count < min_hole_number,hole_count < hole_number,np.max(ice_topo[ponds.astype(bool)]) > pond_bottom_cutoff,hydraulic_head > 0.1*w0,np.mean(ponds) > pc_cutoff)
                    
                    # if t_cutoff > 0, stopping condition is time elapsed >= t_cutoff or ice gets too thin
                    if t_cutoff > 0:
                        cond = (time_current < t_cutoff) and (H_current > 30*dHdt*dt_step)
                else:
                    cond = False
                
                # record variables every time step
                if record_every_step:  
                    hh[-1][-1].append(hydraulic_head)
                    pc[-1][-1].append(np.mean(ponds))
                    time[-1][-1].append(time_current)
                    Nholes[-1][-1].append(hole_count)
                
                # record variables that are used to calculate mean pond evolution
                if record_mean_evolution:   
                                    
                    time_history.append(time_current)
                    Nholes_mean_evolution_history.append(hole_count)
                    hh_mean_evolution_history.append(hydraulic_head)
                    pc_mean_evolution_history.append(np.mean(ponds))
                                
            print('Realization = ' + str(j) + ', Hydraulic head = ' + str(hydraulic_head))
            print('Pond coverage = ' + str(np.mean(ponds)) + ', Pond depth = ' + str(np.mean(water_level[ponds.astype(bool)] - ice_topo[ponds.astype(bool)])))  
            
            # record topography of every realization
            if record_topography:
                ice_topo_record[-1].append(ice_topo)
            
            # record locations of all holes
            if record_every_step:    
                holes_history[-1][-1] = holes             
            
            # record variables at the end of drainage for each realization
            pc_after_drainage[-1][j] = np.mean(ponds)
            hh_after_drainage[-1][j] = hydraulic_head
            number_of_holes[-1][j] = hole_count
            time_elapsed[-1][j] = time_current
            
            # evaluate pond evolution variables at time_eval for each realization
            if record_mean_evolution:   
                
                t_hist = np.hstack([0,time_history,np.inf])
                Nholes_hist = np.hstack([0,Nholes_mean_evolution_history,hole_number])
                hh_hist = np.hstack([w0,hh_mean_evolution_history,hh_mean_evolution_history[-1]])
                pc_hist = np.hstack([1.,pc_mean_evolution_history,pc_mean_evolution_history[-1]])
                
                Nt_interp = interp1d(t_hist,Nholes_hist)
                hht_interp = interp1d(t_hist,hh_hist)
                pct_interp = interp1d(t_hist,pc_hist)
                
                Nholes_mean_evolution_realiztions[j,:] = Nt_interp(time_eval)
                hh_mean_evolution_realiztions[j,:] = hht_interp(time_eval)
                pc_mean_evolution_realiztions[j,:] = pct_interp(time_eval) 

        # calculate mean and std of pond evolution variables over all realizations
        if record_mean_evolution:   
            
            Nholes_mean_evolution[-1] = np.mean(Nholes_mean_evolution_realiztions,axis = 0)
            hh_mean_evolution[-1] = np.mean(hh_mean_evolution_realiztions,axis = 0)
            pc_mean_evolution[-1] = np.mean(pc_mean_evolution_realiztions,axis = 0) 
    
            Nholes_std_evolution[-1] = np.std(Nholes_mean_evolution_realiztions,axis = 0)
            hh_std_evolution[-1] = np.std(hh_mean_evolution_realiztions,axis = 0)
            pc_std_evolution[-1] = np.std(pc_mean_evolution_realiztions,axis = 0)      
        
        index += 1
    
    # collect all variables in Vars into HoleDrain
    collect_variables(Vars,'HoleDrain')
    
    # save HoleDrain into filename
    if save:
        save_Dict(filename,HoleDrain)
