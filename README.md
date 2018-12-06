# Hole model

Python code for the model of Arctic sea ice melt pond drainage through holes that open randomly on a synthetic topography. The code was tested on a 64-bit Mac using python 2.7.10. The model is described in detail in the paper "Critical percolation threshold is an upper bound on Arctic sea ice melt pond coverage" (in preparation).
 
## Description

The full 2d model is contained in the file 2d_hole_model.py. This code will generate pond evolution as more and more holes drain ponds that flood a synthetic surface. The resulting pond evolution is used to estimate the universal function, g(&eta;), so it is needed to run this code at least once before running other scripts that use this function to estimate pond evolution using Eqs. 12 and 14 of the paper. To be able to estimate g(&eta;), pond autocorrelation length is needed, so AstroML should be installed. 2d_hole_model.py initializes a composite run that consits of several individual runs with different parameter combinations. Output of 2d_hole_model.py is a dictionary, HoleDrain, that contains all of the defining parameters of the composite run as well as variables summarizing the resulting pond evolution for each individual run. This dictionary is saved on the hard drive once the code finishes. Details of accessing the variables stored in this dictionary are given within 2d_hole_model.py. 


All of the functions used in 2d_hole_model.py are defined in hole_function_defs.py. There, functions for estimating pond evolution using Eqs. 12 and 14 of the paper are also defined. In particular, pond evolution can be estimated by calling get_pond_evolution_stage_II_III. Additionally, stage I pond evolution described in the Supplementary Material of the paper can also be estimated by calling calc_stageI function defined within hole_function_defs.py. To generate a synthetic topography, call Create_Initial_Topography.  

## Files
2d_hole_model.py
  - Main code for running the full 2d hole model. Library floodfill (https://github.com/cgmorton/flood-fill) must be installed to run this code. Additionally, AstroML (https://github.com/astroML/astroML) should also be installed to calculate the pond autocorrelation function. If it is not installed, the code can still be run by setting record_corr = False

hole_function_defs.py
 - Definitions of specific functions used in all other files. 

plot_synthetic_topography.py
  - A code to generate Fig. 1c and d of the paper.

plot_pi_vs_eta.py
  - A code to generate Fig. 2 of the paper. 2d_hole_model.py should be run before running this file.

plot_2d_vs_estimate.py
  - A code to generate Fig. 3 of the paper. 2d_hole_model.py should be run before running this file.

compare_estimate_and_observations_with_error_bars.py
  - A code to generate Fig. 4a of the paper. Measured pond evolution time-series should be provided (available at http://chrispolashenski.com/home.php).
 
## Dependencies

NumPy, SciPy, Matplotlib, OpenCV, floodfill, astroML
