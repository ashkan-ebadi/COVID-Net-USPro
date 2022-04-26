import os
import numpy as np
import cv2
from scipy import ndimage

# read images 
def read_alphabets(class_directory_path, class_directory_name):
    """
    Reads all the characters/classes from a given alphabet_directory
    """
    datax = []
    datay = []
    images = os.listdir(class_directory_path)
    for img in images:
        if not img.startswith('.') and "convex" in img: 
        # if not img.startswith('.') : 
            # print("reading image:" + img)
            image = cv2.resize(cv2.imread(class_directory_path + '/' + img), (224,224))
            # rotations of image 
            rotated_90 = ndimage.rotate(image, 90)
            rotated_180 = ndimage.rotate(image, 180)
            rotated_270 = ndimage.rotate(image, 270)
            # datax.extend((image, rotated_90, rotated_180, rotated_270))
            # datay.extend((
            #     class_directory_name,
            #     class_directory_name,
            #     class_directory_name,
            #     class_directory_name , 
            # ))
            datax.append(image)
            datay.append(class_directory_name)
    return np.array(datax), np.array(datay)