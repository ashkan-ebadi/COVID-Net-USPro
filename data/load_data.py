import os
import numpy as np
import cv2
from scipy import ndimage
import multiprocessing as mp

# read images in sub directory and rotate 
def read_subdir_image(class_directory_path, class_directory_name):
    """
    Reads all the classes from a given directory
    """
    datax = []
    datay = []
    images = os.listdir(class_directory_path)
    for img in images:
        if not img.startswith('.') and "convex" in img: 
            # print("reading image:" + img)
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

# read all images with multiprocessing
def read_images(base_directory):
    """
    Reads all the alphabets from the base_directory
    Uses multithreading to decrease the reading time drastically
    """
    datax = None
    datay = None
    pool = mp.Pool(mp.cpu_count())

    results = [pool.apply_async(read_subdir_image,
                            args=(
                                base_directory + '/' + directory + '/', directory, )) for directory in os.listdir(base_directory) if not directory.startswith('.')]
    pool.close()

    for result in results:
        if datax is None:
            datax = result.get()[0]
            # print("datax type is:" + type(datax))
            datay = result.get()[1]
        else:
            datax = np.vstack([datax, result.get()[0]])
            datay = np.concatenate([datay, result.get()[1]])
    return datax, datay