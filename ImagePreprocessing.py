import numpy as np
import pandas as pd
import math
import os
from imageio import imread
import matplotlib.pyplot as plt

DIR = '/Users/subra/Desktop/git/Image_Fun/Pictures/'

def resize_image(image, height, width, depth=3):
    image = image[:,:,:depth]
    prev_size = image.shape
    new_image = np.zeros(shape=(height,width,3))
    if prev_size[0]>height:
        mod = int(prev_size[0]/height)+1
        image = image[::mod,:,:]
    if prev_size[1]>width:
        mod = int(prev_size[1]/width)+1
        image = image[:,::mod,:]
    prev_size = image.shape
    new_image[:prev_size[0],:prev_size[1],:] = image
    return new_image.astype(int)

def process_images(path,height=300,width = 300,depth=3):
    image_data = {}
    for image in os.listdir(path):
        print(image)
        image_data[image]=imread(path+image)

    for image in image_data.keys():
        image_data[image] = resize_image(image_data[image],height,width,depth)

    #plt.imshow(image_data['zebra_34.jpg'],interpolation='nearest')
    #plt.show()
    labels = []
    for key in image_data.keys():
        if 'parrot' in key:
            labels.append(1)
        elif 'zebra' in key:
            labels.append(0)
    labels = np.array(labels).reshape(-1,1)
    features = np.array(list(image_data.values())).reshape(-1,height,width,depth)
    return features, labels

if __name__=='__main__':
    features,labels = process_images(DIR)
    print(len(features),len(labels))
