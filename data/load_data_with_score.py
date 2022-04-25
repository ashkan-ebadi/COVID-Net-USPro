import os
import numpy as np
import cv2
from scipy import ndimage
import multiprocessing as mp

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

def read_images_with_score(base_directory, target_vids):
    """
    Reads all the alphabets from the base_directory. 
    Uses multithreading to decrease the reading time drastically. 
    Args:
        base_directory: directory with train or test sets.
        target_vids: names of the videos with targetted LUS score. 
    """
    datax = None
    datay = None
    pool = mp.Pool(mp.cpu_count())

    results = [pool.apply_async(read_subdir_image,
                            args=(
                                base_directory + '/' + directory + '/', directory, target_vids)) for directory in os.listdir(base_directory) if not directory.startswith('.')]
    pool.close()

    for result in results:
        if len(result.get()[0]) != 0: 
            if datax is None:
                datax = result.get()[0]
                datay = result.get()[1]
            else:
                datax = np.vstack([datax, result.get()[0]])
                datay = np.concatenate([datay, result.get()[1]])
    return datax, datay