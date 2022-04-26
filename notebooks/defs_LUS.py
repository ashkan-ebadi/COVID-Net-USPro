import os
import numpy as np
import cv2
from scipy import ndimage

def read_subdir_image(class_directory_path, class_directory_name, target_vid_names):
    """
    Load image data from a  given a directory. 
    Args:
        class_directory_path: directory with images separated into class. Each class is a different folder.
        class_directory_name: name of the classes. 
        target_vid_names: names of the videos with targetted LUS score. 
    """
    datax = []
    datay = []
    images = os.listdir(class_directory_path)
    for img in images:
        contains_in_target_vid = len([string for string in target_vid_names if img.startswith(string)]) != 0
        if not img.startswith('.') and "convex" in img and contains_in_target_vid: 
            image = cv2.resize(cv2.imread(class_directory_path + '/' + img), (224,224))
            # rotations of image 
            rotated_90 = ndimage.rotate(image, 90)
            rotated_180 = ndimage.rotate(image, 180)
            rotated_270 = ndimage.rotate(image, 270)
            datax.extend((image, rotated_90, rotated_180, rotated_270))
            datay.extend((
                class_directory_name,
                class_directory_name,
                class_directory_name,
                class_directory_name , 
            ))
    return np.array(datax), np.array(datay)