import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Dense,Flatten,Dropout,Activation
import numpy as np
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

def process_images(path):
    image_data = {}
    for image in os.listdir(path):
        print(image)
        image_data[image]=imread(path+image)

    for image in image_data.keys():
        print(image)
        image_data[image] = resize_image(image_data[image],300,300)

    #plt.imshow(image_data['zebra_34.jpg'],interpolation='nearest')
    #plt.show()
    labels = []
    for key in image_data.keys():
        if 'parrot' in key:
            labels.append(1)
        elif 'zebra' in key:
            labels.append(0)
    features = image_data.values()
    return features,labels

def get_convnet():
    model = Sequential()

    #looks for more global features
    model.add(Conv2D(150,(6,6),activation='leaky_relu',input_shape=(300,300,3)))
    model.add(MaxPooling2D(pool_size = (3,3)))
    #looks for more local features
    model.add(Conv2D(75,(4,4),activation = 'leaky_relu'))
    model.add(MaxPooling2D(pool_size = (3,3)))
    #looks for extremely local features
    model.add(Conv2D(30,(2,2),activation = 'leaky_relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    #Flattening layer
    model.add(Flatten())
    #Fully connected layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    #sigmoid output layer
    model.add(Dense(1,activation = 'sigmoid'))

    model.compile(loss = 'binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

if __name__=='__main__':
    features, labels = process_images(DIR)
    model = get_convnet
