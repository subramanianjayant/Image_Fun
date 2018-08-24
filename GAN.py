#Fix compilation bug

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ImagePreprocessing as ipp

height = 128
width = 128
depth = 3
DIR =  '/Users/subra/Desktop/git/Image_Fun/Pictures/'

features,labels = ipp.process_images(DIR,height=height,width=width,depth=depth)
batch_size = len(features)

#discriminator (CNN classifier)
def discriminator():
    model = Sequential()

    model.add(Conv2D(32, 5, strides=3,input_shape=(height,width,depth),
        padding='same',activation=LeakyReLU(alpha = 0.2)))
    model.add(Conv2D(32, 5, strides=2,input_shape=(height,width,depth),
        padding='same',activation=LeakyReLU(alpha = 0.2)))
    model.add(Conv2D(64, 5, strides=1,input_shape=(height,width,depth),
        padding='same',activation=LeakyReLU(alpha = 0.2)))
    model.add(Flatten())
    model.add(Dense(64, activation = LeakyReLU(alpha=0.2)))
    model.add(Dense(1,activation = 'sigmoid'))
    return model

#generator (deconvolutions) takes noise vectors of size 128
def generator():
    dim = 16
    depth = 1

    model = Sequential()

    model.add(Dense(dim*dim*depth, input_dim = 128, activation = LeakyReLU(alpha=0.2)))
    model.add(BatchNormalization(momentum=0.99))
    model.add(Reshape((dim,dim,depth)))
    model.add(Conv2DTranspose(144, 5, strides=2,padding = 'same'
        ,activation = LeakyReLU(alpha=0.2)))
    model.add(BatchNormalization(momentum=0.99))
    model.add(Conv2DTranspose(72, 5, strides=2,padding = 'same'
        ,activation = LeakyReLU(alpha=0.2)))
    model.add(BatchNormalization(momentum=0.99))
    model.add(Conv2DTranspose(24, 5, strides=2,padding = 'same'
        ,activation = LeakyReLU(alpha=0.2)))
    model.add(BatchNormalization(momentum=0.99))
    model.add(Conv2DTranspose(3, 5, strides=1,padding = 'same'
        ,activation = 'sigmoid'))
    return model

def disc_model():
    optimizer = RMSprop(lr=0.00005)
    dm = discriminator()
    dm.compile(loss = 'binary_crossentropy',optimizer = optimizer,metrics=['accuracy'])
    return dm

def adversarial_model():
    optimizer = RMSprop(lr = 0.00005)
    am = Sequential()
    am.add(generator())
    am.add(discriminator())
    am.compile(loss = 'binary_crossentropy',optimizer = optimizer,metrics=['accuracy'])
    return am

gen = generator()
discriminator = disc_model()
adversarial = adversarial_model()

for iter in range(20):
    noise = np.random.randint(0,256, size = [batch_size, 128])
    images_fake = gen.predict(noise)
    plt.imshow(images_fake[0], interpolation='nearest')
    plt.show()

    x = np.concatenate((features, images_fake))
    #1 is real, 0 is fake
    y = np.ones([2*batch_size,1])
    y[batch_size:,:] = 0
    disc_loss = discriminator.train_on_batch(x,y)

    y = np.ones([batch_size, 1])
    noise = np.random.randint(0,256, size = [batch_size, 128])
    adv_loss = adversarial.train_on_batch(noise, y)

    log_mesg1 = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
    log_mesg2 = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
    print('iteration %s' %(iter))
    print(log_mesg1)
    print(log_mesg2)
    print('\n')
