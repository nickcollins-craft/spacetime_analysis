"""
This is a code to perform a Fourier Transform analysis of the spacetime plots 
produced by the space_time_analyser code. This will tell us the velocity of the
particles at a given radial point
"""

# Import the libraries that we will need
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack,misc,optimize
from matplotlib.patches import Ellipse

# Set the import folder
import_folder = "/home/nick/Bureau/Photos/Essai_2018.04.25/analysis/spacetimes/"
# Get the radius list
radius_list = np.load("/home/nick/Bureau/Photos/Essai_2018.04.25/analysis/greyscale_values/radius_list.npy")

radius = radius_list[0]

# Get the image that we want
image_name = import_folder + "spacetime_radius_" + str(radius) +"_increment_ 1.png"
spacetime = misc.imread(image_name,mode='L')

# Form the window function along each axis, before multiplying them together to get the window for the 2D matrix
"""theta_window_hann = np.hanning(spacetime.shape[0])
time_window_hann = np.hanning(spacetime.shape[1])
window_2D_hann =np.outer(theta_window_hann,time_window_hann)
theta_window_hamming = np.hamming(spacetime.shape[0])
time_window_hamming = np.hamming(spacetime.shape[1])
window_2D_hamming =np.outer(theta_window_hamming,time_window_hamming)

# Apply the window function to the spacetime image
windowed_spacetime_hann = np.multiply(spacetime,window_2D_hann)
windowed_spacetime_hamming = np.multiply(spacetime,window_2D_hamming)"""

# Get the 2D Fourier transform
im_fft = fftpack.fft2(spacetime)
"""im_fft_hann = fftpack.fft2(windowed_spacetime_hann)
im_fft_hamming = fftpack.fft2(windowed_spacetime_hamming)"""
# Shift the zero-component to the centre
im_fft = fftpack.fftshift(im_fft)
"""im_fft_hann = fftpack.fftshift(im_fft_hann)
im_fft_hamming = fftpack.fftshift(im_fft_hamming)"""
# Get the absolute values of the transform
im_fft_abs = np.abs(im_fft)
"""im_fft_abs_hann = np.abs(im_fft_hann)
im_fft_abs_hamming = np.abs(im_fft_hamming)"""

# We also try setting a threshold to the FFTed image, discarding anything below a certain fraction of the maximum value.
# The threshold is only applied after the logarithm stage so that we don't try to find log(0)
threshold_val = 0.75

# We now take some steps so that we can plot more clearly
# Put it in the logarithmic space
im_fft_abs_log = np.log10(im_fft_abs)
#im_fft_abs_thresholded_truths = im_fft_abs_log > threshold_val*np.nanmax(im_fft_abs_log)
#im_fft_abs_thresholded_log = im_fft_abs_log*im_fft_abs_thresholded_truths
"""im_fft_abs_hann_log = np.log10(im_fft_abs_hann)
im_fft_abs_hamming_log = np.log10(im_fft_abs_hamming)"""
# Find the minimum and maximum plus the range
lowest = np.nanmin(im_fft_abs_log)
highest = np.nanmax(im_fft_abs_log)
contrast_range = highest - lowest
#lowest_thresholded = np.nanmin(im_fft_abs_thresholded_log)
#highest_thresholded = np.nanmax(im_fft_abs_thresholded_log)
#contrast_range_thresholded = highest_thresholded - lowest_thresholded
"""lowest_hann = np.nanmin(im_fft_abs_hann_log)
highest_hann = np.nanmax(im_fft_abs_hann_log)
contrast_range_hann = highest_hann - lowest_hann
lowest_hamming = np.nanmin(im_fft_abs_hamming_log)
highest_hamming = np.nanmax(im_fft_abs_hamming_log)
contrast_range_hamming = highest_hamming - lowest_hamming"""
# Normalise the image
normed_fft = 255*(im_fft_abs_log-lowest)/contrast_range
#normed_fft_thresholded = 255*(im_fft_abs_thresholded_log-lowest_thresholded)/contrast_range_thresholded
"""normed_fft_hann = 255*(im_fft_abs_hann_log-lowest_hann)/contrast_range_hann
normed_fft_hamming = 255*(im_fft_abs_hamming_log-lowest_hamming)/contrast_range_hamming

# Plot the image
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0,0].set_title('Unmodified')
axes[0,0].imshow(normed_fft,cmap='gray')
axes[0,1].set_title('Thresholded at ' + str(threshold_val)+ ' of log10(max)')
axes[0,1].imshow(normed_fft_thresholded,cmap='gray')
axes[1,0].set_title('Hann Window')
axes[1,0].imshow(normed_fft_hann,cmap='gray')
axes[1,1].set_title('Hamming Window')
axes[1,1].imshow(normed_fft_hamming,cmap='gray')
plt.tight_layout()
plt.show()"""

# Test a range of threshold values to see the effect on our elipse
threshold_values = np.arange(0.6,0.8,0.01)
theta_values = np.zeros(len(threshold_values))
theta_plot_values = np.zeros(len(threshold_values))
major_axis_values = np.zeros(len(threshold_values))
minor_axis_values = np.zeros(len(threshold_values))
gradient_values = np.zeros(len(threshold_values))
fig, axes = plt.subplots(4,5)
for (threshold_val,threshold_index) in zip(threshold_values,range(len(threshold_values)-1)):
    
    im_fft_abs_thresholded_truths = im_fft_abs_log > threshold_val*np.nanmax(im_fft_abs_log)
    im_fft_abs_thresholded_log = im_fft_abs_log*im_fft_abs_thresholded_truths
    lowest_thresholded = np.nanmin(im_fft_abs_thresholded_log)
    highest_thresholded = np.nanmax(im_fft_abs_thresholded_log)
    contrast_range_thresholded = highest_thresholded - lowest_thresholded
    normed_fft_thresholded = 255*(im_fft_abs_thresholded_log-lowest_thresholded)/contrast_range_thresholded
    # We want to find the boundaries of the area with the points that we actually care about, based on the thresholded version
    upper_left_corner = (np.min(np.nonzero(im_fft_abs_thresholded_log)[0]),np.min(np.nonzero(im_fft_abs_thresholded_log)[1]))
    lower_right_corner = (np.max(np.nonzero(im_fft_abs_thresholded_log)[0]),np.max(np.nonzero(im_fft_abs_thresholded_log)[1]))
    # We get the submatrix that contains the information that we care about
    fitting_matrix = im_fft_abs[upper_left_corner[0]:lower_right_corner[0],upper_left_corner[1]:lower_right_corner[1]]
    fitting_matrix_binary = im_fft_abs_thresholded_truths[upper_left_corner[0]:lower_right_corner[0],upper_left_corner[1]:lower_right_corner[1]]
    fitting_matrix_normed = normed_fft_thresholded[upper_left_corner[0]:lower_right_corner[0],upper_left_corner[1]:lower_right_corner[1]]
    
    # Get the coordinates
    y_coords = range(fitting_matrix.shape[0])
    x_coords = range(fitting_matrix.shape[1])
    # Get the appropriate moments
    Area = float(np.sum(np.outer(np.power(y_coords,0),np.power(x_coords,0))*fitting_matrix*fitting_matrix_binary))
    Mx1 = float(np.sum(np.outer(np.power(y_coords,0),np.power(x_coords,1))*fitting_matrix*fitting_matrix_binary))
    My1 = float(np.sum(np.outer(np.power(y_coords,1),np.power(x_coords,0))*fitting_matrix*fitting_matrix_binary))
    Mxy1 = float(np.sum(np.outer(np.power(y_coords,1),np.power(x_coords,1))*fitting_matrix*fitting_matrix_binary))
    x_centre = Mx1/Area
    y_centre = My1/Area
    Mx2 = float(np.sum(np.outer(np.power(y_coords,0),np.power(x_coords,2))*fitting_matrix*fitting_matrix_binary))
    My2 = float(np.sum(np.outer(np.power(y_coords,2),np.power(x_coords,0))*fitting_matrix*fitting_matrix_binary))
    central_x2 = Mx2/Area - x_centre**2
    central_xy = Mxy1/Area - x_centre*y_centre
    central_y2 = My2/Area - y_centre**2
    # Calculate the orientation angle of the major axis
    theta_values[threshold_index] = 0.5*np.arctan((2*central_xy)/(central_x2-central_y2))
    theta_plot_values[threshold_index] = np.degrees(theta_values[threshold_index])
    major_axis_values[threshold_index] = (6*(central_x2+central_y2+(4*central_xy**2+(central_x2-central_y2)**2)**0.5))**0.5
    minor_axis_values[threshold_index] = (6*(central_x2+central_y2-(4*central_xy**2+(central_x2-central_y2)**2)**0.5))**0.5
    
    # Perform a linear regression fit
    lin_x_coords = np.zeros(np.sum(im_fft_abs_thresholded_truths))
    lin_y_coords = np.zeros(np.sum(im_fft_abs_thresholded_truths))
    point_counter = 0
    # Get the x and y values of the points which clear the threshold
    for y_index in range(fitting_matrix_binary.shape[0]):
        for x_index in range(fitting_matrix_binary.shape[1]):
            if fitting_matrix_binary[y_index,x_index]:
                lin_x_coords[point_counter] = x_index - fitting_matrix.shape[1]/2
                lin_y_coords[point_counter] = y_index - fitting_matrix.shape[0]/2
                point_counter = point_counter + 1
    # Define the linear function to fit
    def lin_fit_func(x_data,gradient):
       return gradient*x_data
    # Return the value of the function that works the best
    optimal_gradient, covariance = optimize.curve_fit(lin_fit_func,lin_x_coords,lin_y_coords,0)
    gradient_values[threshold_index] = optimal_gradient[0]
    optimum_fit = gradient_values[threshold_index]*lin_x_coords
    x_data_for_plot = lin_x_coords + fitting_matrix.shape[1]/2
    y_data_for_plot = optimum_fit + fitting_matrix.shape[0]/2
        
    # Plot the ellipse
    ells = Ellipse((x_centre,y_centre),major_axis_values[threshold_index],minor_axis_values[threshold_index],theta_plot_values[threshold_index],edgecolor='red',facecolor='none')
    axes.flat[threshold_index].add_artist(ells)
    axes.flat[threshold_index].imshow(fitting_matrix_normed,cmap='gray')
    axes.flat[threshold_index].plot(x_data_for_plot,y_data_for_plot)
    axes.flat[threshold_index].set_title('Threshold value = ' + format(threshold_val,'.4f'))

# Have a look at the images so we can choose a good value
plt.show()

# Calculate the slope of the ellipses from the angle
ellipse_slopes = np.tan(theta_values)
velocity_ellipses = -1/ellipse_slopes
velocity_line_of_best_fit = -1/gradient_values