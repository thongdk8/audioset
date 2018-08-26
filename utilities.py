import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from skimage.color import gray2rgb
import cv2
import imageio

def list_files(path):
    # files = [os.path.join(path, file) for file in os.listdir(dir) if (file[-4:] == '.png' or file[-4:] == '.jpg')]
    files = []
    file_id = []
    for file in os.listdir(path):
        if (file[-4:] == '.wav' or file[-4:] == '.mp3' or file[-4:] == '.amr'):
            files.append(os.path.join(path, file))
            file_id.append(file[:-4])
    return files, file_id


def compute_class_weights(dataset_dir):
    total = 0
    for root, dirs, files in os.walk(dataset_dir):
        total += len(files)

    class_weights = {}
    classes = np.sort([c for c in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, c))] )
    #print(classes)
    for i, c in enumerate(classes):
        n_files = len( os.listdir(os.path.join(dataset_dir,str(c))) )
        class_weights[i] = total/(len(classes) * n_files)
    
    return class_weights

def count_image_samples(dataset_dir):
    total = 0
    for root, dirs, files in os.walk(dataset_dir):
        total += len(files)
    return total+100
