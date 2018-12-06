import numpy as np
import cv2
from scipy import ndimage
from scipy.interpolate import interp1d
import os.path
import pickle
from scipy.special import erf, erfinv
from scipy.integrate import ode
import scipy.stats as stats
from scipy.stats import multivariate_normal
import floodfill
from astroML.correlation import two_point 

"""
Specific functions used to calculate pond drainage are defined here.
"""

"""
A function to load a dictionary. Filename should contain the full path to the 
dictionary and version should be an integer.
"""
def load_Dict(filename,version):
        
    if version > 0:
        filename += '_v' + str(version)
        
    with open(filename + '.pickle', 'rb') as handle:
        Dict = pickle.load(handle)
        
    return Dict


"""
A function to save a dictionary as a .pickle file. Filename should contain the full path to the 
dictionary; dictionary name, Dict, should be given as a string; if version < 0, save_Dict will not
overwrite previous files. If version >=0 and a file with that version already exists, save_Dict 
will ask the user whether to overwrite the existing file.
"""
def save_Dict(filename,Dict,version = -1):
    if version < 0:
        version = 0
        if not(os.path.isfile(filename + '.pickle')):
            with open(filename + '.pickle', 'wb') as handle:
                pickle.dump(Dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:    
            version = 1
            
            fname = filename
            while os.path.isfile(fname+'.pickle'):  
                fname = filename + '_v' + str(version)
                version += 1
            print('Version = ' + str(version-1))
    
            with open(fname + '.pickle', 'wb') as handle:
                pickle.dump(Dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if version > 0:
            fname = filename + '_v' + str(version)
        else: 
            fname = filename
        if not(os.path.isfile(filename + '.pickle')):
            with open(fname + '.pickle', 'wb') as handle:
                pickle.dump(Dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        else:
            ans = input('Overwrite exiting file v%i? (y/n): '%version)
            if ans == 'y':
                with open(fname + '.pickle', 'wb') as handle:
                    pickle.dump(Dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    return version


"""
A function to create a synthetic topography. Returns a numpy array of size (res,res).
    res - size of the output array
    mode - topography type. Options - 'snow_dune', 'diffusion', and 'rayleigh'
    tmax - time to diffuse a random configuration in 'diffusion' and 'rayleigh' topographies. Controls the typical length-scale
    dt - time-step for diffusion in 'diffusion' and 'rayleigh' topographies. If too large, creating a topography may fail
    g - anisotropy parameter
    sigma_h - standard deviation of the topography
    h - mean elevation
    snow_dune_radius - mean radius of mounds in the 'snow_dune' topography. Controls the typical length-scale
    Gaussians_per_pixel - density of mounds in the 'snow_dune' topography (number of mounds * snow_dune_radius^2 / res^2)
    number_of_r_bins - number of categories of mound radii to consider in the 'snow_dune' topography
    window_size - cutoff parameter for placing mounds in the 'snow_dune' topography
    snow_dune_height_exponent - exponent that relates mound radius and mound height in the 'snow_dune' topography 
"""

def Create_Initial_Topography(res = 500, mode = 'snow_dune',tmax = 2,dt = 0.1, g = 1,sigma_h = 1., h = 0., snow_dune_radius = 1., Gaussians_per_pixel = 0.2, 
                              number_of_r_bins = 150, window_size = 5, snow_dune_height_exponent = 1.):
                              

    if mode == 'diffusion':
        t = np.arange(0,tmax,dt)
        ice_topo = 0.5-np.random.rand(res,res)
        stencil = np.array([[0, g, 0],[1, -2*(1+g), 1], [0, g, 0]])
        
        for i in range(1,len(t)):
            ice_topo += dt*ndimage.convolve(ice_topo, stencil)
            
    if mode == 'rayleigh':
        t = np.arange(0,tmax,dt)
        ice_topo1 = 0.5-np.random.rand(res,res)
        ice_topo2 = 0.5-np.random.rand(res,res)
        stencil = np.array([[0, g, 0],[1, -2*(1+g), 1], [0, g, 0]])
        
        for i in range(1,len(t)):
            ice_topo1 += dt*ndimage.convolve(ice_topo1, stencil)
            ice_topo2 += dt*ndimage.convolve(ice_topo2, stencil)
        
        ice_topo = np.sqrt(ice_topo1**2+ice_topo2**2)
        
    if mode == 'snow_dune':
        ice_topo = np.zeros([res,res])
        N = np.ceil((res/snow_dune_radius)**2 * Gaussians_per_pixel).astype(int)
        r0 = np.random.exponential(snow_dune_radius,N)
        
        bins = np.linspace(np.min(r0),np.max(r0),number_of_r_bins+1)
        r0_bins = np.zeros(number_of_r_bins)
        r0_N = np.zeros(number_of_r_bins).astype(int)
        
        for i in range(1,number_of_r_bins):
            loc = (r0 >= bins[i-1]) & (r0 < bins[i])
            r0_bins[i] = np.mean(r0[loc])
            r0_N[i] = np.sum(loc)

        r0_bins = r0_bins[r0_N>0] 
        r0_N = r0_N[r0_N>0] 
        
        for i in range(len(r0_bins)):
            r = r0_bins[i]
            h0 = r**snow_dune_height_exponent / snow_dune_radius**snow_dune_height_exponent
            cov = np.eye(2); cov[1,1] = g
            cov *= r**2
            
            rv = multivariate_normal([0,0], cov) 
            
            x0 = np.random.choice(np.arange(-res/2,res/2),r0_N[i])
            y0 = np.random.choice(np.arange(-res/2,res/2),r0_N[i])
        
            x = np.arange(-np.ceil(r*window_size).astype(int),np.ceil(r*window_size).astype(int)+1)
            y = x.copy()
            X,Y = np.meshgrid(x,y)
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X; pos[:, :, 1] = Y
            
            G = rv.pdf(pos) * 2 * np.pi * np.sqrt(np.linalg.det(cov)) * h0
            
            for j in range(r0_N[i]):
                loc_x = ((pos[:,:,0]+x0[j]) % res).astype(int)
                loc_y = ((pos[:,:,1]+y0[j]) % res).astype(int)
        
                ice_topo[loc_x,loc_y] += G
        
    ice_topo /= np.std(ice_topo)
    ice_topo -= np.mean(ice_topo)
    ice_topo *= sigma_h
    ice_topo += h
    return ice_topo      

"""
A function to find autocorrelation function. AstroML (https://github.com/astroML/astroML) needs to be installed to work.
    ponds should as a boolean array with True corresponding to ponded locations
    random_fraction can be used to speed up find_corr by only considering a fraction of all ponded points
    bins are distance bins at which autocorrelation is evaluated

Returns estimated autocorrelation function, distances at which autocorrelation was estimated, and autocorrelation length, l0
"""

def find_corr(ponds,random_fraction,bins):
    random_points = np.random.rand(np.shape(ponds)[0],np.shape(ponds)[1]) < random_fraction
    ponds_coordinates = np.array(np.nonzero(ponds*random_points)).T
    corr = two_point(ponds_coordinates, bins)
    loc = corr == corr
    
    C = corr[loc]
    d = bins[1:][loc]

    C = np.hstack([1.,C,0.])
    d = np.hstack([0.,d,np.inf])
    
    Cd_interp = interp1d(C,d,kind = 'linear')
    l0 = Cd_interp(np.exp(-1))
    
    return_corr = np.hstack([1.,corr,0.])
    return_dist = np.hstack([0.,bins[1:],np.inf])
    return return_corr,return_dist,l0

"""
Melt ponded locations by decreasing topography elevation by delta_t * DH / Tmelt. Returns updated topography.
"""

def melt(ice_topo,ponds,delta_t,T_melt,DH):
    return ice_topo - delta_t/T_melt*ponds*DH

"""
Drain ponds through holes. Drainage proceeds until all holes are either above water level or ponds are at sea level.
floodfill (https://github.com/cgmorton/flood-fill) has to be installed in order to work.
    
    holes - coordinates of all holes. Should be supplied as a (N,2) numpy array of integers from 0 to res-1
    ice_topo - float array of size (res,res). Ice topography
    ponds - boolean array of size (res,res) marking the locations of ponds defined as water_level > ice_topo
    water_level - float array of size (res,res). Each connected pond has a constant water level and non-ponded regions have water_level = ice_topo. Ponds are water_level > ice_topo
    sea_level - sea level height (float). Should be set to 0
    DH - ponded ice melt rate is determined as dh_diff/dt = DH/Tmelt (float). Only relevant if separate_timescales = False.
    DH_drain -  depth of water drained during Tdrain (float)
    H - ice thickness (float). Used to maintain hydrostatic balance
    dt - drainage time step (float)
    Tdrain - time to drain water of depth DH_drain (float)
    Tmelt - ponded ice melt rate is determined as dh_diff/dt = DH/Tmelt (float). Only relevant if separate_timescales = False.
    rho_w - water density
    rho_i - ice density
    conn - used to determine pond connectivity. Can be 4 or 8
    separate_timescales - if True, pond drainage occurs separately from preferential ponded ice melt. Otherwise, drainage and melting occur at the same time
    hydrostatic_adjustment - if True, maintain hydrostatic balance
    eps - cutoff to acertain if ponds are at sea level
    
Returns updated topography, water level, ponds, pond coverage and time elapsed drianing.
"""            

def drain(holes,ice_topo,ponds,water_level,sea_level,DH,DH_drain,H,dt,Tdrain,Tmelt,rho_w = 1000.,rho_i = 900.,conn = 8, separate_timescales = True, hydrostatic_adjustment = True, eps = 1E-12):

    count = 0
    pc = []
    
    while True:
        
        ret, L = cv2.connectedComponents(ponds,connectivity = conn)    
         
        sea_level_holes = (water_level[holes[:,0],holes[:,1]] <= sea_level + eps).astype(bool)
        ice_holes = (water_level[holes[:,0],holes[:,1]] <= ice_topo[holes[:,0],holes[:,1]] + eps).astype(bool)
        active_holes = np.ones(len(holes[:,0]),dtype = bool) - (sea_level_holes + ice_holes)
        hole_labels = L[holes[:,0],holes[:,1]]
                
        if np.sum(active_holes) > 0:
            active_ponds = np.array(list(set(np.unique(hole_labels[active_holes])) - {0}))
            active_ponds_mask = (np.in1d(L,active_ponds).reshape(np.shape(L)[0],np.shape(L)[1])).astype(bool)
        else:
            break
        
        delta_h = dt/Tdrain*DH_drain
        dh_fill = np.zeros(np.shape(L))
        outflow_pts = []
        
        for active_pond in active_ponds:
            current_holes = np.nonzero(hole_labels == active_pond)[0]    
            depth = np.min(np.vstack([water_level[holes[current_holes,0],holes[current_holes,1]] - sea_level, \
                         water_level[holes[current_holes,0],holes[current_holes,1]] - ice_topo[holes[current_holes,0],holes[current_holes,1]]]),axis = 0)
            delta_w_idx = np.argmax(depth)
            delta_w = depth[delta_w_idx]
            dh = np.max([np.min([delta_w,delta_h]),0.])
            dh_fill[holes[current_holes[delta_w_idx],0],holes[current_holes[delta_w_idx],1]] = dh
            outflow_pts.append(holes[current_holes[delta_w_idx],:])
        
        dh_fill[~active_ponds_mask] = np.nan
        dh_fill = floodfill.from_points(dh_fill, outflow_pts, four_way=False)
        dh_fill[~active_ponds_mask] = 0.
        
        water_level -= dh_fill
        d_w = 0; ponds_at_sea_level_mask = np.array([1.]); ponds_at_sea_level = []
        
        if (hydrostatic_adjustment):
            
            if np.sum(sea_level_holes) > 0:
                ponds_at_sea_level = np.array(list(set(np.unique(L[holes[sea_level_holes,0],holes[sea_level_holes,1]])) - {0}))
                ponds_at_sea_level_mask = (1-(np.in1d(L,ponds_at_sea_level).reshape(np.shape(L)[0],np.shape(L)[1]))).astype(np.uint8)
            else: 
                ponds_at_sea_level_mask = np.ones(np.shape(L),dtype = np.uint8)
            
            d_w = ( (rho_w - rho_i)/rho_w * H - np.mean(water_level) )/np.mean(ponds_at_sea_level_mask)
            
            ice_topo += d_w
            water_level[ponds_at_sea_level_mask.astype(bool)] += d_w            
        
        ponds = (ice_topo < water_level).astype(np.uint8)
        ice = (1-ponds).astype(bool)
        water_level[ice] = ice_topo[ice]
        
        if separate_timescales == False:
            ice_topo =  melt(ice_topo,ponds,dt,Tmelt,DH)
        pc = np.hstack([pc,np.mean(ponds)])
        
        count += 1
        
        sea_level_holes = ( water_level[holes[:,0],holes[:,1]] <= sea_level + dt/Tdrain*DH_drain ).astype(bool)
        ice_holes = ( water_level[holes[:,0],holes[:,1]] <= ice_topo[holes[:,0],holes[:,1]]  + dt/Tdrain*DH_drain ).astype(bool)
        active_holes = np.ones( len(holes[:,0]),dtype = bool ) - ( sea_level_holes + ice_holes )
        
        if np.sum(active_holes) == 0:
            break
    
    time_elapsed = count*dt           
    return ice_topo,water_level,ponds,pc,time_elapsed

"""
Find cumulative distribution at x for several probability density functions defined in pdf. param controls parameters of these distributions
"""
def cdf(x,param,pdf):
    if pdf == 'normal':
            return 0.5*(1+erf(x/np.sqrt(2)))
    if pdf == 'exp':
        return np.exp(x)
    if  pdf == 'logistic':
        return 1./(1.+np.exp(-x))
    if pdf == 'weibul':
        return np.exp( - (-x)**param[0] )

"""
Find quantile function at x for several probability density functions defined in pdf. param controls parameters of these distributions
"""  
def quant(x,param,pdf):
    if pdf == 'normal':
        return np.sqrt(2)*erfinv(2.*x-1.)
    if pdf == 'exp':
        return np.log(x)
    if  pdf == 'logistic':
        return np.log(x/(1-x))
    if pdf == 'weibul':
        return - ( -np.log(x) )**(1./param[0])       

"""
Find renormalized number of holes, eta, at time t

    p - pond coverage
    N0 - total number of possible holes
    l0 - typical pond length-scale
    L - size of the domain
    t - time at which to evaluate
    Th - hole opening timescale
    t0 - center of the hole opening distribution
    c - topographic constant in eta
    pdf - hole opening distribution
    param - parameters of the distribution
    calc_t0 - if True, t0 is estimated using Eq. 9 of the paper. Otherwise, t0 that is supplied is used
    include_p - used for convenience if t is interpreted as the memorization timescale that depends on p. If True, t is divided by (1-p). 
    
"""  
def find_N_normal(p,N0,l0,L,t,Th,t0 = 0,c=1.,pdf = 'normal',param = [1],calc_t0 = True,include_p = True):
    if include_p:
        t /= (1-p)
    if calc_t0:
        t0 = - quant(1./N0, param, pdf) * Th
    return float(N0)*float(l0)**2/float(L)**2 * c * cdf( (t - t0)/Th, param, pdf)
        

"""
Find renormalized pond coverage, pi, at time t accordin to Eq. 12 of the paper. Works iteratively until a self-consistent solution is found
up to tolerance eps. If t is interpreted as the memorization timescale, Tm, as in Eqs. 13 and 14 of the paper, include_p should be set to True

Most parameters have the same meaning as in find_N_normal. 
    gx - universal function g(eta). Shouuld be supplied as a callable function
    pthersh - the percolation threhsold 

"""  
def find_ppc(gx, N0, l0, L, t, Th, pthresh, t0 = 0, c=1., pdf = 'normal', param = [1], calc_t0 = True, include_p = True, eps = 1E-10):
    
    eps = 1E-10
    p = 0.5; p1 = 0; p2 = 1.
    Nx = find_N_normal(p = p*pthresh, N0 = N0, l0 = l0, L = L ,t = t, Th = Th, t0 = t0, c=1., pdf = pdf, param = param, calc_t0 = calc_t0, include_p = include_p)
            
    while np.abs(p - gx(Nx)) > eps:
        if p - gx(Nx) >= 0:
            p2 = p
        else:
            p1 = p
        p = 0.5*(p1 + p2)
        Nx = find_N_normal(p = p*pthresh, N0 = N0, l0 = l0, L = L ,t = t, Th = Th, t0 = t0, c=1., pdf = pdf, param = param, calc_t0 = calc_t0, include_p = include_p) #find_N_normal(p*pthresh,N0,l0,res,Tm,Thole,pdf,param)
    return p

"""
Solve ordinary differential equation, Eq. 4 of the paper, to estimate the 
universal function g(eta). 

""" 
def g_ode(dg = 0.001,gmin = 0.0001):
    g = 1.- dg
    eta = 1.E-10

    gs = []
    etas = []
    while g > gmin:
        gs.append(g)
        etas.append(eta)

        del_g = g**2 * (1-g)**(-19./18.)
        g -= dg
        eta += dg/del_g

    gs = np.array(gs)
    etas = np.array(etas)

    return gs,etas

"""
Get the universal function g(eta).

If version >= 0, g(eta) is based on a previously saved run of the full 2d model. This run should contain no 
ponded ice melt (Tmelt = np.inf) and no ice thinning (include_ice_thinning = False). It should also have the mean 
evolution recorded (record_mean_evolution = True). A callable interpolation of the simulated pi = pi(eta) will be returned
along with the points x which were used to make the interpolation. A portion of simulated pi = pi(eta) above the percolation
threshold has to be cut off or find_ppc will not converge. This is done by setting xmin as a lower cutoff on eta and 
interpolation is performed on eta > xmin, while for eta < xmin, it is assumed that pi increases linearly to 1 at eta = 0. 
An appropriate values for the percolation threshold and the constant c should be supplied in pthresh and c. 

    filename - path to previously saved HoleDrain
    version - version to load
    idx - combination of parameters to choose within given HoleDrain
    
If version < 0, g(eta) is estimated by solving ordinary deifferential equation, Eq. 4 of the paper by calling g_ode.
The solution to this equation is calibrated to the drainage curve on the snow dune topography by multiplying the 
resulting eta by c['ode'], set by default to 0.29 based on our previous calibration.
""" 
def get_gx(filename,version = 0,idx = 0, xmin = 0.001, pthresh = {'diffusion': 0.5, 'rayleigh': 0.4, 'snow_dune': 0.46}, c = {'diffusion': 1.2, 'rayleigh': 1., 'snow_dune': 1., 'ode': 0.29} ):
    
    if version >= 0:
        HoleDrain = load_Dict(filename,version)
    
        res = HoleDrain['Record_list'][idx]['res']
        mode = HoleDrain['Record_list'][idx]['mode']
        l0_c = np.mean(HoleDrain['l0'][idx])
    
        xh = HoleDrain['Nholes_mean_evolution'][idx]*l0_c**2/res**2 * c[mode]
        yh = np.array(HoleDrain['pc_mean_evolution'][idx])/pthresh[mode]
        
        del HoleDrain
    
    else:
        yh,xh = g_ode()
        xh *= c['ode']
    
    xmax = np.max(xh)
    loc = (xh > xmin) & (xh < xmax)
    
    x = np.hstack([0,xh[loc],100,np.inf])
    y = np.hstack([1,yh[loc],0,0])
    
    gx = interp1d(x,y)
    
    return x,gx

"""
Calculate pond evolution according to Eqs. 12 to 14 of the paper. Timescales Th and Tm can either be estimated 
using physical parameters by setting use_physical_params = True, or can be supplied directly if use_physical_params = False.
Physical parameters should be supplied in a dictionary physical_params with all of the parameters in units of m, s, kg, C, and J.
If physical parameters are supplied, ice thinning rate, dH/dt, is calculated based on fluxes and pond coverage. Otherwise,
a thinning rate has to be supplied directly. Center of the hole opening distribution, t0, can either be supplied directly if 
estimate_t0 = False or can be estimated according to Eq. 9 of the paper if estimate_t0 = True. If version >= 0, filename, version, and idx
provide path, version, and a particular run of the 2d hole model used to interpolate the universal function, g(eta). If version < 0, Eq. 4 of the paper is
used to estimate g(eta). Estimate is started at a time t_initial and pond coverage pinit. Evolution between pinit and the percolation threhsold, pthresh,
is represented as a straight line between (t_initial,pinit) and (C,pthresh). Pond coverage is evaluated
at tsteps times between t_initial and tmax. At each time step between t_initial+Tdrain and tmax, pond coverage is 
estimated as the maximum between pond coverage evaluated using Eq. 12 of the paper and pond coverage evaluated using Eq. 14 of the paper.  
Returns time in days, pond coverage, and location of times between t_initial and t_initial + Tdrain.

"""
day = 24.*3600.
physical_params_def = { 'F' : 254., 'ai' : 0.6, 'ap' : 0.25, 'T0' : 1.2, 'z' : 0.6, 'S' : 3.,
            'c' : 6.5, 'kappa' : 1.5, 'k' : 1.8, 'DeltaT' : 0.7, 'Fr' : -25.,'gamma' : 1.8E4,
            'Lm' : 3.006E8, 'rho_w' : 1000., 'rho_i' : 900.}

def get_pond_evolution_stage_II_III(filename,Tm = 360.*3600, Th = 120.*3600, Tdrain = 0.*day, t0 = 9.*day, N0 = 2.25E8, pthersh = 0.33, c=1., l0 = 3.3, L = 1500., H = 1.5,
    dHdt = 3./100./day, version = 0, idx = 0, estimate_t0 = True, use_physical_params = True, physical_params = physical_params_def, \
    tsteps = 100, t_initial = 0.*day, tmax = 60.*day, pdf = 'normal', param = [2.], pinit = 0.33):
    
    if use_physical_params:
        F = physical_params['F']; ai = physical_params['ai']; 
        ap = physical_params['ap']; T0 = physical_params['T0'];
        z = physical_params['z']; S = physical_params['S']; c = physical_params['c']; kappa = physical_params['kappa']; k = physical_params['k'];
        DeltaT = physical_params['DeltaT']; Fr = physical_params['Fr']; gamma = physical_params['gamma'];
        Lm = physical_params['Lm']; rho_w = physical_params['rho_w']; rho_i = physical_params['rho_i']
        
        da = ai-ap
        Tm = Lm/(da*F)*(rho_w-rho_i)/rho_w*H
        Th = DeltaT / (T0**2/(rho_i*gamma*S) * (c*k*T0/H**2 + (1-ap) * F * kappa * np.exp(-kappa*z) ))                 
    
    
    time = np.linspace(t_initial,tmax,tsteps)
    dt = (time[1] - time[0])
    p = np.zeros(tsteps)
    
    if estimate_t0:
        t0 = -quant(1./N0, param, pdf)*Th
    
    x,gx = get_gx(filename,version,idx)
    
    pII = np.zeros(tsteps)
    pIII = np.zeros(tsteps)
    
    H_current = H
    loc_drain = np.zeros(tsteps).astype(bool)
    for j in range(tsteps):
        
        if time[j] - t_initial <= Tdrain:
            p_trial = pthersh*find_ppc(gx = gx,N0 = N0,l0 = l0,L = L,t = Tdrain,Th = Th,pthresh = pthersh,t0 = t0,c = c,pdf = pdf,\
                        param = param,calc_t0 = False,include_p = False,eps = 1E-10)
            
            pII[j] = pinit - (pinit - p_trial)*(time[j] - t_initial)/Tdrain 
            loc_drain[j] = True        
        
        else:
            pII[j] = pthersh*find_ppc(gx = gx,N0 = N0,l0 = l0,L = L,t = (time[j] - t_initial),Th = Th,pthresh = pthersh,t0 = t0,c = c,pdf = pdf,\
                        param = param,calc_t0 = False,include_p = False,eps = 1E-10)
            
            
        if use_physical_params:
            F_melt = p[j]*(1-ap)*F + (1-p[j])* (1-ai)*F + Fr
            dH = dt*F_melt/Lm
            H_current -= dH
            Tmelt = Lm / (da*F) * (rho_w-rho_i)/rho_w * H_current
        else:
            H_current = H - (time[j] - t_initial)*dHdt
            Tmelt = Tm / H * H_current
        pIII[j] = pthersh*find_ppc(gx = gx,N0 = N0,l0 = l0,L = L,t = Tmelt,Th = Th,pthresh = pthersh,t0 = t0,c = c,pdf = pdf,\
                    param = param,calc_t0 = False,include_p = True,eps = 1E-10)
            
    
    p = np.max(np.vstack([pII,pIII]),axis = 0)  
    return time/day,p,loc_drain          
   

"""
Calculate pond evolution using the solution derived in the Supplementary Material of the paper. A dictionary of
parameters should be supplied. All the parameters should be in unites of m, s, J, and kg. This dictionary should contain:
    
    bare snow melt rate, 'dhs_dt'
    ponded snow melt rate, 'dhps_dt'
    ponded ice melt rate, 'dhpi_dt'
    water density, 'rho_w'
    ice density, 'rho_i'
    snow density, 'rho_s'
    domain size, 'L'
    percolation threshold, 'pc'
    volumetric drainage, 'Q'
    default ponded ice fraction, 'pi'
    method of calculation, 'method'
    snow depth distribution, 'f_dist'
    
If method is 'simple', calculations are performed according to Eqs S24 of the Supplementary Material. Otherwise
calculations according to Eqs. S38 are performed. Drainage is assumed to only occur beyond the percolation threshold,
according to Eq. S39. Default ponded ice coverage fraction, pi, only has an effect if method is not 'simple'. As there is 
no systematic way to choose this number, it should be set to 0. f_dist should be a callable probability density function.
By default, it should be set to a Gamma distribution. Returns time array, t, and the solution, sol. sol[:,0] represents pond coverage,
while sol[:,1] represents water level.
"""

def calc_stageI(params):
    
    def f(t,y,params):
        
        hs = params['dhs_dt']
        hps = params['dhps_dt']
        hpi = params['dhpi_dt']
        
        rho_w = params['rho_w']
        rho_i = params['rho_i']
        rho_s = params['rho_s']
        
        L = params['L']
        Q = params['Q']
        pc = params['pc']
            
        pi = params['pi']
        
        ri = rho_i/rho_w
        rs = rho_s/rho_i
        
        f_dist = params['f_dist']
        method = params['method']
        p = y[0]; w = y[1]
        
        if method == 'simple':
            dwdt = (ri*(1-p)*hs  - Q * np.max([0,p/pc-1]) / L**2 / rs)/(1./rs-1+p) 
        else:
            dwdt = (ri*(1-p)*hs - (1-ri)* ( hps*p + pi*(hpi/rs - hps) ) - Q*np.max([0,p/pc-1]) / L**2 / rs  )/(1./rs-1+p)
        dpdt = f_dist(w+hs*t) * (dwdt + hs)
        return np.array([dpdt,dwdt])
        
    day = 3600.*24; hour = 3600
    t1 = 20*day; dt = 1.*hour
    y0, t0 = np.array([0,0]), 0
    
    r = ode(f).set_integrator('vode', method='bdf')
    r.set_initial_value(y0, t0).set_f_params(params)
    
    t = np.array([t0]); sol = y0.copy()
    while r.successful() and r.t < t1:
        t = np.hstack([t,r.t+dt])
        sol = np.vstack([sol,r.integrate(r.t+dt)])
        
    return t, sol

"""
Find areas and perimeters of disconnected ponds.
    Im - a binary 2d array that contains ponds
    mpp - meters per pixel. Used to convert areas and perimeters in pixels to physical units

Returns areas and perimeters of each pond
"""

def FindAreaPerimeter(Im,mpp):
    
    Im = np.lib.pad(Im, (1,1), 'constant')	       
    ret, L = cv2.connectedComponents(Im)
                          
    L_sort = np.sort(L[L >0])
    a = np.nonzero(np.hstack([1,np.diff(L_sort),1]))[0]
    As = (np.diff(a))
                            
    C = np.zeros(np.shape(L),dtype = int)
    n = (4.*Im - (np.roll(Im,1,axis = 0)+np.roll(Im,-1,axis = 0)+np.roll(Im,1,axis = 1)+np.roll(Im,-1,axis = 1)))*Im
                        
    C[n>0] = L[n>0]   
    nh = np.zeros(np.shape(n)); nh[n>0] = 1
                
    n -= (1.-np.sqrt(2)/2) * (np.roll(np.roll(nh,1,axis = 0),1,axis = 1)+np.roll(np.roll(nh,1,axis = 0),-1,axis = 1)\
            +np.roll(np.roll(nh,-1,axis = 0),1,axis = 1)+np.roll(np.roll(nh,-1,axis = 0),-1,axis = 1))    
            
    loc = C > 0
    indcs = np.argsort(C[loc])
                
    C_sort = C[loc][indcs]
    n_sort = n[loc][indcs]
    
    positions = np.nonzero(np.hstack([1,np.diff(C_sort),1]))[0]

    Ps = np.zeros(len(positions)-1)
    for j in range(len(positions)-1):
        Ps[j] = np.sum(n_sort[positions[j]:positions[j+1]])
        
    As = As * mpp**2
    Ps = Ps * mpp
        
    return As, Ps
    
     