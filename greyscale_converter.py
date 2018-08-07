"This is a code to transform the images of ACSA from RGB to an 8-bit greyscale"
# Import the relevant functions
from PIL import Image

# Give the first and last images (needs to be actual last image + 1)
first_image = 4
last_image = 11524

for i in range(first_image,last_image):
    
    # Assign string form to the value of i
    if i < 10:
        image_number = "000" + str(i)
    elif i < 100:
        image_number = "00" + str(i)
    elif i < 1000:
        image_number = "0" + str(i)
    else:
        image_number = str(i)
        
    #Give the names of the images to import and export
    import_image = "/home/nick/Bureau/Photos/Essai_2018.04.26/original/IMG_" + image_number + ".JPG"
    export_image = "/home/nick/Bureau/Photos/Essai_2018.04.26/greyscale/img_" + image_number + ".jpg"
    
    # Operate on the image (open, crop, greyscale, save)
    im = Image.open(import_image)
    im=im.crop(box=(0,270,3888,2592))
    im = im.convert("L")
    im.save(export_image)