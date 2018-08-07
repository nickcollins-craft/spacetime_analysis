"""
This is a code to perform a space-time analysis for the data given by the 
arc_length_analyser code. We have the goal of extracting the velocity which can
be further analysed to give us a distribution with the radius
"""

# Import the packages and functions that we need 
import numpy as np
import matplotlib.pyplot as plt

# Set the time elapsed between each photo
dt = 5
#
# Set the import and export folders
import_folder = "/home/nick/Bureau/Photos/Essai_2018.04.25/analysis/greyscale_values/"
export_folder = "/home/nick/Bureau/Photos/Essai_2018.04.25/analysis/spacetimes/"

# Import the list of radii we want to work with
radius_list = np.load(import_folder + "radius_list.npy")

# For each radius in radius list, get the associated greyscale file
for radius in radius_list:
    # Create the import filename and get the file
    import_filename = import_folder + "radius_" + str(radius) + ".npy"
    greyscale_matrix = np.load(import_filename)
    
    # Extract the coordinates to know the theta and time values
    theta_coordinates = greyscale_matrix[0,0,:]
    time_coordinates = np.arange(0,dt*greyscale_matrix.shape[0]+1,dt)
    
    # Create a matrix with dimensions (#time slices)x(# thetas) that contains the greyscale values
    plotting_matrix = greyscale_matrix[:,1]
    plotting_matrix = plotting_matrix.transpose()
    
    # We also want to try and force our matrix to be of a size equal to a power of 2 in both directions
    # We determine the power of 2 that is below the size of the matrix
    N = 0
    while 2**N < plotting_matrix.shape[0]:
        N = N + 1
    N = N - 1
    # We cut the plotting matrix to size, maintaining symmetry
    plotting_matrix = plotting_matrix[(plotting_matrix.shape[0]-2**N)/2:plotting_matrix.shape[0]-(plotting_matrix.shape[0]-2**N)/2,:]
    
    
    # Save that spacetime bad boi
    filename = export_folder + "spacetime_radius_" + str(radius) + ".png"
    plt.imsave(filename,plotting_matrix,cmap='gray')
    #plt.savefig(filename,bbox_inches='tight',pad_inches=0)
    
    """Hamming windows"""
    """Give a weight to the value of the greyscale when we do the best fit"""
    """Talk with Sheng about FFT and windowing etc"""