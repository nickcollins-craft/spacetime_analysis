"""
This is a code to run through the outputs of arc_length_analyser and stitch 
them together to provide one full time-series for a given radius
"""

# First we import the relevant packages and functions
import numpy as np
import os

# We obtain the relevant lists of radii and number of time blocks to account for
import_folder = "/home/nick/Bureau/Photos/Essai_2018.04.25/analysis/greyscale_values/"
export_folder = "/home/nick/Bureau/Photos/Essai_2018.04.25/analysis/greyscale_values/"

# We get the radius list
radius_list = np.load(import_folder + "radius_list.npy")
# We get the number of time blocks
time_block_number = np.load(import_folder + "time_list.npy")

# We go through each radius in the list of radii
for radius in radius_list:
    # We go through each time block
    for block_number in time_block_number:
        # We load in the block
        blockname = import_folder + "radius_" + str(radius) + "_timeblock_" + str(block_number) + ".npy"
        greyscale_block = np.load(blockname)
        
        # We find how big the total greyscale matrix needs to be by loading in the first image
        if block_number == 0:
            # Dimension the full greyscale matrix accordingly
            greyscale = np.zeros((greyscale_block.shape[0],greyscale_block.shape[1],greyscale_block.shape[2]))
            greyscale[:,:,:] = greyscale_block[:,:,:]
        else:
            greyscale = np.append(greyscale,greyscale_block,axis=0)
        
        
        # We delete the block files for cleanliness
        os.remove(import_folder + "radius_" + str(radius) + "_timeblock_" + str(block_number) + ".npy")
    
    # We save the radius file
    radial_filename = export_folder + "radius_" + str(radius) + ".npy"
    np.save(radial_filename,greyscale)
    
# We remove the time address list for cleanliness
os.remove(import_folder + "time_list.npy")