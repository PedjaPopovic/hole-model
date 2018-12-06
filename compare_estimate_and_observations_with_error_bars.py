import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from hole_function_defs import *

"""
This is the code to generate Fig. 4a of the paper "Critical percolation threshold 
is an upper bound on Arctic sea ice melt pond coverage." To run this code, universal function g(eta)
should be estimated either by first running 2d_hole_model and saving the HoleDrain dictionary or by 
solving Eq. 4 of the paper by setting version_gx < 0. Data from Polashenski et al. are 
available at http://chrispolashenski.com/home.php
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


# initialize variables
pinit = l0_m = L_m = N0_m = F_m = ai_m = ap_m = pd_m = H_m = T0_m = z_m = S_m = \
c_m = kappa = k_m = DeltaT_m = pc_m = Fr = n0_m = tmax = t_initial = Tdrain_m = 0

filename_gx = 'Path/to/HoleDrain'   # Path to HoleDrain dictionary used to evaluate g(eta)
version_gx = -1                      # Version of HoleDrain to load to evaluate g(eta). Set version_gx < 0 if a HoleDrain dictionary was not previously saved.
idx = 0                             # choose combination of parameters in the 2d hole model run to estimate g(eta)

filename_data = 'Path/to/Data.txt'  # Path to pond evolution time-series measurements

N_trials = 100         # number of parameter combinations
rel_err_low  = 0.1      # relative error of the well-constrained physical parameters
rel_err_high  = 0.1     # relative error of the poorly-constrained physical parameters

pdf = 'normal'          # assumed hole opening distribution
param = [2.]            # parameters of the hole opening distribution

day = 24.*3600.         # number of seconds in a day

# latent heat of melting, water density, ice density, constant that relates salinity to heat capacity
l = 3.006E8; rhow = 1000.; rhoi = 900.; gamma = 1.8E4;  

# dictionary of assumed mean physical parameters
Param_dict = {'pinit': 0.53, 'l0_m': 3.3, 'L_m': 1500., 'n0_m': 100., \
                      'F_m': 254., 'ai_m': 0.63, 'ap_m': 0.25, 'pd_m': 0.25, 'H_m': 1.2, \
                      'T0_m' : 1.2, 'z_m' : 0.6, 'S_m': 3., 'c_m' : 6.5, 'kappa' : 1.5, 'k_m' : 1.8, 'DeltaT_m' : 0.7,
                      'pc_m' : 0.35, 'Fr': -25., 't_initial': 5., 'tmax': 25., 'Tdrain_m': 2.}

# get python variables from dictionary keys                    
get_variables("Param_dict")

N0_m = n0_m*L_m**2;         # mean number of brine channels derived from mean brine channel density and domain size
tsteps = 100                # number of times at which to evaluate pond coverage according to Eqs. 12 and 14 of the paper
t_initial = t_initial*day   # initial time in seconds
tmax = tmax*day             # final time in seconds

time = np.linspace(t_initial,tmax,tsteps)
dt = (time[1] - time[0])
p = np.zeros([N_trials,tsteps])

"""
All parameters are drawn N_trials times from a normal distribution with means
defined in Param_dict and standard deviations equal to mean * rel_error.  
"""
l0 = np.random.normal(loc = l0_m, scale = rel_err_low*l0_m, size = N_trials)
L = np.random.normal(loc = L_m, scale = rel_err_low*L_m, size = N_trials)
N0 = np.random.normal(loc = N0_m, scale = rel_err_low*N0_m, size = N_trials)

F = np.random.normal(loc = F_m, scale = rel_err_low*F_m, size = N_trials)
ai = np.random.normal(loc = ai_m, scale = rel_err_low*ai_m, size = N_trials)
ap = np.random.normal(loc = ap_m, scale = rel_err_low*ap_m, size = N_trials)
pd = np.random.normal(loc = pd_m, scale = rel_err_low*pd_m, size = N_trials)
H = np.random.normal(loc = H_m, scale = rel_err_low*H_m, size = N_trials)

T0 = np.random.normal(loc = T0_m, scale = rel_err_high*T0_m, size = N_trials)
z = np.random.normal(loc = z_m, scale = rel_err_high*z_m, size = N_trials)
S = np.random.normal(loc = S_m, scale = rel_err_low*S_m, size = N_trials)
c = np.random.normal(loc = c_m, scale = rel_err_high*c_m, size = N_trials)
k = np.random.normal(loc = k_m, scale = rel_err_low*k_m, size = N_trials)
DeltaT = np.random.normal(loc = DeltaT_m, scale = rel_err_high*DeltaT_m, size = N_trials)
Tdrain = np.random.normal(loc = Tdrain_m, scale = rel_err_low*Tdrain_m, size = N_trials)*day

pc = np.random.normal(loc = pc_m, scale = rel_err_low*pc_m, size = N_trials)

# Draw centers of the hole opening distribution from a Gumbel distribution
Th = DeltaT / (T0**2/(rhoi*gamma*S) * (c*k*T0/H**2 + (1-ap) * F * kappa * np.exp(-kappa*z) ))
t0 = np.random.gumbel(loc = -quant(1./N0,param,pdf), scale = 0.25)*Th
  
# Load pond evolution time-series data
PolashEvol = np.loadtxt(filename_data,delimiter=',')
PolashTime = PolashEvol[:,0]
PolashPonds = PolashEvol[:,1]

# Interpolate observations to compare with estimates and choose the best fit
Polash_interp = interp1d(PolashTime,PolashPonds)

# Create N_trials estimates
p = np.zeros([N_trials,tsteps])
Loss = np.zeros(N_trials)
for i in range(N_trials):
    
    print('trial: ' + str(i))
    physical_params = { 'F' : F[i], 'ai' : ai[i], 'ap' : ap[i], 'T0' : T0[i], 'z' : z[i], 'S' : S[i],
            'c' : c[i], 'kappa' : kappa, 'k' : k[i], 'DeltaT' : DeltaT[i], 'Fr' : Fr,'gamma' : gamma,
            'Lm' : l, 'rho_w' : rhow, 'rho_i' : rhoi}
    t,p_est, _ = get_pond_evolution_stage_II_III(filename = filename_gx, Tm = 0., Th = 0., Tdrain = Tdrain[i], \
                                             t0 = t0[i], N0 = N0[i],  pthersh = pc[i], c = 1., l0 = l0[i], L = L[i], \
                                             H = H[i], dHdt = 0, version = version_gx, idx = idx, \
                                             estimate_t0 = False, use_physical_params = True, physical_params = physical_params, \
                                             tsteps = tsteps, t_initial = t_initial, tmax = tmax, pdf = pdf, param = param, pinit = pinit)

    
    p[i,:] = p_est
    loc = (t < np.max(PolashTime)) & (t > t[0]+Tdrain[i]/day)
    
    # Record mean squared error of the estimate relative to data
    Loss[i] = np.mean((p[i,:][loc]-Polash_interp(time[loc]/day))**2)

# Chose the estimated pond evolution that fits the data best
i_best = np.argmin(Loss)
p_best = p[i_best,:]

# Calculate the mean and standard deviation of estimated pond evolution
p_mean = np.mean(p,axis = 0)
p_std = np.std(p,axis = 0)
p_up = p_mean+p_std
p_down = np.max(np.vstack([np.zeros(len(p_mean)),p_mean-p_std]),axis = 0)

plt.figure(1); # Compare data to measurements
plt.clf(); 

plt.plot(time/day,p_mean,'r--',lw = 3)
plt.plot(PolashTime,PolashPonds,'go-',lw = 2)

plt.plot(time/day,p_best,'k--',lw = 1)
plt.fill_between(time/day, p_down, p_up,facecolor='red',alpha = 0.3)

plt.xlabel('Time [days]',fontsize = 18)
plt.ylabel('p',fontsize = 18)
plt.xlim(0,25.)
plt.ylim(0,0.8)
plt.legend(['Prediction using g(x)', 'Measurements'],frameon = False)

plt.show()