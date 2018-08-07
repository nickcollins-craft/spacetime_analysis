"""
This is a code to analyse along particular arc lengths, extracting the values
of greyscale along particular radius for a particular arc length, and saving
them for analysis in a different code
"""

# Import the necessary functions
from PIL import Image
import numpy as np
import imageio

# Specify the assumed pixel coordinates of the centre (they're outside the image)
x_centre = 3850/2  # Trial and error so don't touch
y_centre = 3320  # Trial and error so don't touch
# Specify the list of radii to look in
radius_list = range(1950, 3250, 50)

# Specify the angle we want our averaging arc to subtend at the smallest radius
subtended_angle_smallest_radius = np.deg2rad(30)
arc_length = radius_list[0]*subtended_angle_smallest_radius

# Give the folder to import images from
import_folder = "/home/nick/Bureau/Photos/Essai_2018.04.25/greyscale/img_"
# Give the first and last images
first_image = 4
last_image = 11519
# How long do we want the time blocks we cut it up into to be (this is just a trick we will use to manage RAM)?
time_block_length = 5
# Create a vector with the list of block endpoints
time_block_ends_vector = range(first_image+time_block_length, last_image+1, time_block_length)
if time_block_ends_vector[-1] < last_image:
    time_block_ends_vector = np.append(time_block_ends_vector, [last_image])

# Give the folder to export data to
export_folder = "/home/nick/Bureau/Photos/Essai_2018.04.25/analysis/greyscale_values/"

# Set to display mode or analysis mode
display = False
analysis = 1 - display

"""# Look into the parallax error correction (a function of height?)
# Look into if jpg is the best option to work with
# Look into distortion functions in say ImageJ"""

# We write a block that can work out and display the radii that we are checking along. It is set to on/off
if display:
    for i in range(first_image, first_image):
        # Assign string form to the value of i
        if i < 10:
            image_number = "000" + str(i)
        elif i < 100:
            image_number = "00" + str(i)
        elif i < 1000:
            image_number = "0" + str(i)
        else:
            image_number = str(i)

        # Give the names of the images to import and export
        import_image = import_folder + image_number + ".jpg"

        # Get the image
        im = Image.open(import_image)
        # Convert it to RGB space
        im = im.convert('RGB')
        # Convert it to a numpy array
        im = np.array(im)
        # Turn the points that are on the circle red
        for y in range(im.shape[0]):
            for x in range(im.shape[1]):
                for radius in radius_list:
                    # Get the points that are either exactly on the circle, or ones that have fractional values
                    if (x-x_centre)**2 + (y-y_centre)**2 == radius**2:
                        im[y, x, 0] = 255
                    elif ((x-1)-x_centre)**2 + (y-y_centre)**2 >= radius**2 and ((x+1)-x_centre)**2 + (y-y_centre)**2 <= radius**2:
                        im[y, x, 0] = 255
                    elif ((x-1)-x_centre)**2 + (y-y_centre)**2 <= radius**2 and ((x+1)-x_centre)**2 + (y-y_centre)**2 >= radius**2:
                        im[y, x, 0] = 255
                    elif (x-x_centre)**2 + ((y-1)-y_centre)**2 >= radius**2 and (x-x_centre)**2 + ((y+1)-y_centre)**2 <= radius**2:
                        im[y, x, 0] = 255
                    elif (x-x_centre)**2 + ((y-1)-y_centre)**2 <= radius**2 and (x-x_centre)**2 + ((y+1)-y_centre)**2 >= radius**2:
                        im[y, x, 0] = 255
        # Convert it back into an image
        im = Image.fromarray(im, 'RGB')
        im.show()

# Write the code for the analysis section
if analysis:
    # Get the first image
    if first_image < 10:
        im = imageio.imread(import_folder + "000" + str(first_image) + ".jpg")
    elif first_image < 100:
        im = imageio.imread(import_folder + "00" + str(first_image) + ".jpg")
    elif first_image < 1000:
        im = imageio.imread(import_folder + "0" + str(first_image) + ".jpg")
    else:
        im = imageio.imread(import_folder + str(first_image) + ".jpg")

    # We create a matrix that stores the x,y,r,theta values. We declare it bigger than we need for efficiency
    initial_length = 210000
    information_matrix = np.zeros((4, initial_length))
    # Create a point counter that increments so we can save the list of points
    point_counter = 0
    # Create a vector that tells us where the radius changes
    radius_change_vector = np.zeros(len(radius_list), int)

    # We work out the region of interest as a function of each radius
    for (radius, rad_index) in zip(radius_list, range(len(radius_list))):
        # We work out the width of the image that we are looking in, based on the desired arc length
        subtended_angle = arc_length/radius
        x_left = int(x_centre + np.ceil(radius*np.cos((np.pi/2)+(subtended_angle/2))))
        x_right = int(x_centre + np.ceil(radius*np.cos((np.pi/2)-(subtended_angle/2))))
        # We work out the height of the image that we are looking in, from the centre line radius, and the bottom point of the radius we are considering
        y_top = int(y_centre - radius)
        y_bottom = int(y_centre - np.floor(radius*np.sin((np.pi/2)-(subtended_angle/2))))

        for y in range(y_top, y_bottom):
            for x in range(x_left, x_right):
                # Create a boolean variable that will tell us afterwards if we're on the circle
                on_circle = False
                # Get the points that are either exactly on the circle, or ones that have fractional values
                if (x-x_centre)**2 + (y-y_centre)**2 == radius**2:
                    on_circle = True
                elif ((x-1)-x_centre)**2 + (y-y_centre)**2 >= radius**2 and ((x+1)-x_centre)**2 + (y-y_centre)**2 <= radius**2:
                    on_circle = True
                elif ((x-1)-x_centre)**2 + (y-y_centre)**2 <= radius**2 and ((x+1)-x_centre)**2 + (y-y_centre)**2 >= radius**2:
                    on_circle = True
                elif (x-x_centre)**2 + ((y-1)-y_centre)**2 >= radius**2 and (x-x_centre)**2 + ((y+1)-y_centre)**2 <= radius**2:
                    on_circle = True
                elif (x-x_centre)**2 + ((y-1)-y_centre)**2 <= radius**2 and (x-x_centre)**2 + ((y+1)-y_centre)**2 >= radius**2:
                    on_circle = True

                # Now add it to the list of coordinates to check if true
                if on_circle:
                    information_matrix[0, point_counter] = x
                    information_matrix[1, point_counter] = y
                    information_matrix[2, point_counter] = radius
                    if x == x_centre:
                        information_matrix[3, point_counter] = np.pi/2
                    else:
                        # Use tan^-1 so both x and y are accounted for
                        information_matrix[3, point_counter] = np.arctan2((y_centre-y), (x-x_centre))
                    point_counter = point_counter + 1

        # Update that we are finished with this radius and the vector will change here
        radius_change_vector[rad_index] = point_counter

    # We shrink the information matrix slightly if it has some zeroes at the end
    information_matrix = information_matrix[0:4, 0:point_counter]

    # Now we create the loop that runs through the time-blocks
    for (working_last_image, time_block_index) in zip(time_block_ends_vector, range(len(time_block_ends_vector))):
        # Create the grey-scale matrix and fill the already known entries appropriately
        greyscale = np.zeros((time_block_length, 5, information_matrix.shape[1]))
        greyscale[:, 0, :] = information_matrix[0, :]
        greyscale[:, 1, :] = information_matrix[1, :]
        greyscale[:, 2, :] = information_matrix[2, :]
        greyscale[:, 3, :] = information_matrix[3, :]
        if time_block_index == 0:
            working_first_image = first_image
        else:
            working_first_image = time_block_ends_vector[time_block_index-1]

        # Now we need to write a code that extracts the grey-scale values from the original at the index positions we want to check.
        for (image_index, t) in zip(range(working_first_image, working_last_image), range(time_block_length)):

            # Assign string form to the value of image_index
            if image_index < 10:
                image_number = "000" + str(image_index)
            elif image_index < 100:
                image_number = "00" + str(image_index)
            elif image_index < 1000:
                image_number = "0" + str(image_index)
            else:
                image_number = str(image_index)

            # Declare the string for the image to import
            import_image = import_folder + image_number + ".jpg"
            # Import the image
            im = imageio.imread(import_image)

            # Now a loop that looks at the relevant points
            for point_index in range(greyscale.shape[2]):
                greyscale[t, 4, point_index] = im[int(greyscale[t, 1, point_index]), int(greyscale[t, 0, point_index])]

        # Save files of the greyscale values characterised by theta and time. Each radial value has its own file
        for (radius, rad_index) in zip(radius_list, range(len(radius_list))):
            # Access the desired part of the greyscale matrix. We get the theta values and the grey scale values for each time
            if rad_index == 0:
                radial_matrix = greyscale[:, 3:5, 0:radius_change_vector[rad_index]]
            else:
                radial_matrix = greyscale[:, 3:5, radius_change_vector[rad_index-1]+1:radius_change_vector[rad_index]]

            # Now we re-order the list of points by ascending values of theta
            sorting_array = np.argsort(radial_matrix, axis=2)[0, 0, :]
            sorted_radial_matrix = radial_matrix[:, :, sorting_array]
            # Create the file name
            export_file = export_folder + "radius_" + str(radius) + "_timeblock_" + str(time_block_index) + ".npy"
            # Save the numpy array in numpy format
            np.save(export_file, sorted_radial_matrix)

    # Save the radius list and number of time blocks as files so the code to produce the images knows where to look
    radius_save_file = export_folder + "radius_list.npy"
    time_save_file = export_folder + "time_list.npy"
    np.save(radius_save_file, radius_list)
    np.save(time_save_file, range(len(time_block_ends_vector)))
