import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Dense,Flatten,Dropout,Activation
from keras.callbacks import EarlyStopping
from keras import optimizers
import numpy as np
import ImagePreprocessing as ipp
import matplotlib.pyplot as plt

DIR = '/Users/subra/Desktop/git/Image_Fun/Pictures/'

def get_convnet():
    model = Sequential()

    #looks for more global features
    model.add(Conv2D(16,(5,5),strides=3, activation='relu',input_shape=(300,300,3)))
    model.add(MaxPooling2D(pool_size = (2,2)))
    #looks for more local features
    model.add(Conv2D(32,(4,4),strides = 2, activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    #looks for extremely local features
    model.add(Conv2D(64,(3,3),activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    #Flattening layer
    model.add(Flatten())
    #Fully connected layer
    model.add(Dense(64, activation='relu'))
    #sigmoid output layer
    model.add(Dense(1,activation = 'sigmoid'))

    sgd = optimizers.SGD(lr = 0.00005, momentum = 0.9, decay = 0, nesterov=True)

    model.compile(loss = 'binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model

def train_model(model,features_train,labels_train,val_size=0.2):
    stop = EarlyStopping(monitor='val_loss',patience=5)
    model.fit(features_train,labels_train,epochs = 50,callbacks=[stop]
        ,batch_size = 20,validation_split=val_size)
    return model

def test_model(model,features_test,labels_test):
    print(model.evaluate(features_test,labels_test))

model = get_convnet()
features,labels = ipp.process_images(DIR)

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size = 0.2)


#plt.imshow(features_train[5], interpolation='nearest')
#plt.show()
model = train_model(model,features_train,labels_train)
#model.save('convnet.h5')
#del model

#model = load_model('convnet.h5')
test_model(model,features_test,labels_test)
