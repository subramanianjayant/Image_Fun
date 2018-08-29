#architecture inspired by
#https://towardsdatascience.com/implementing-a-generative-adversarial-network-gan-dcgan-to-draw-human-faces-8291616904a

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ImagePreprocessing as ipp

height = 128
width = 128
depth = 3
DIR =  '/Users/subra/Desktop/git/Image_Fun/Pictures/'

features,labels = ipp.process_images(DIR,height=height,width=width,depth=depth)

tf.reset_default_graph()
batch_size = len(features)
n_noise = 64

X_in = tf.placeholder(dtype=tf.float32,shape=[None,height,width,depth],name='X')
noise = tf.placeholder(dtype=tf.float32, shape=[None,n_noise])
is_training = tf.placeholder(dtype=tf.bool,name='is_training')

def leaky_relu(x):
    return tf.maximum(x,tf.multiply(x,0.2))

def binary_cross_entropy(x, z):
    eps = 1e-12
    return (-(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps)))

#discriminator (CNN classifier)
def discriminator(input,reuse=None):
    height,width,depth = input.shape[1:]
    with tf.variable_scope("discriminator",reuse=reuse):
        clf = tf.reshape(input,shape=[-1,height,width,depth])
        clf = tf.layers.conv2d(clf,kernel_size=5,filters=32,strides=3
            ,padding='same',activation = leaky_relu)
        clf = tf.layers.conv2d(clf,kernel_size=5,filters=32,strides=2
            ,padding='same',activation = leaky_relu)
        clf = tf.layers.conv2d(clf,kernel_size=5,filters=64,strides=1
            ,padding='same',activation = leaky_relu)
        clf = tf.contrib.layers.flatten(clf)
        clf = tf.layers.dense(clf,units=64,activation=leaky_relu)
        clf = tf.nn.dropout(clf,keep_prob=0.7)
        clf = tf.layers.dense(clf,units=1,activation=tf.nn.sigmoid)
        return clf

#generator (deconvolutions)
def generator(vec,is_training = is_training):
    with tf.variable_scope("generator",reuse=None):
        gen=vec
        dim=8
        depth=1
        gen = tf.layers.dense(gen, units=dim*dim*depth, activation = leaky_relu)
        gen = tf.contrib.layers.batch_norm(gen, is_training=is_training, decay=0.99)
        gen = tf.reshape(gen, shape = [-1,dim,dim,depth])
        gen = tf.image.resize_images(gen, size = [16,16])
        gen = tf.layers.conv2d_transpose(gen,kernel_size=5,filters=144,strides=2
            ,padding='same',activation=leaky_relu)
        gen = tf.contrib.layers.batch_norm(gen, is_training=is_training, decay=0.99)
        gen = tf.layers.conv2d_transpose(gen,kernel_size=5,filters=72,strides=2
            ,padding='same',activation=leaky_relu)
        gen = tf.contrib.layers.batch_norm(gen, is_training=is_training, decay=0.99)
        gen = tf.layers.conv2d_transpose(gen,kernel_size=5,filters=24,strides=2
            ,padding='same',activation=leaky_relu)
        gen = tf.contrib.layers.batch_norm(gen, is_training=is_training, decay=0.99)
        gen = tf.layers.conv2d_transpose(gen,kernel_size=5,filters=3,strides=1
            ,padding='same',activation=tf.nn.sigmoid)
        return gen

gen = generator(noise, is_training=is_training)
disc_true = discriminator(X_in)
disc_false = discriminator(gen, reuse=True)

#variables
gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
disc_vars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]

#regularization
#disc_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l1_regularizer(1e-6), disc_vars)
#gen_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l1_regularizer(1e-6), gen_vars)

#losses
loss_disc_true = binary_cross_entropy(tf.ones_like(disc_true),disc_true)
loss_disc_false = binary_cross_entropy(tf.zeros_like(disc_false),disc_false)
loss_gen = tf.reduce_mean(binary_cross_entropy(tf.ones_like(disc_false),disc_false))
loss_disc = tf.reduce_mean(0.5*(loss_disc_true+loss_disc_false))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_disc = tf.train.RMSPropOptimizer(learning_rate = 0.00015).minimize(loss_disc, var_list=disc_vars)
    optimizer_gen = tf.train.RMSPropOptimizer(learning_rate = 0.00008).minimize(loss_gen, var_list=gen_vars)

session = tf.Session()
session.run(tf.global_variables_initializer())

#training GAN
for iter in range(1,101):
    train_disc = True
    train_gen = True

    n = np.random.randint(0,256,[batch_size,n_noise]).astype(np.float32)
    batch = [np.reshape(img, [height,width,depth]) for img in features]

    #prints a generated image
    if iter%25==0:
        generated_images = session.run(gen, feed_dict={noise: n, is_training:False})
        plt.imshow(generated_images[0], interpolation='nearest')
        plt.show()

    print('computing losses')
    d_real_ls, d_fake_ls, gen_ls, d_ls = session.run([loss_disc_true, loss_disc_false, loss_gen, loss_disc],
        feed_dict={X_in: batch, noise: n, is_training:True})

    d_real_ls = np.mean(d_real_ls)
    d_fake_ls = np.mean(d_fake_ls)
    gen_ls = gen_ls
    d_ls = d_ls
    print('iteration %d | discriminator_loss = %f\t generator_loss = %f'
        %(iter,d_ls,gen_ls))

    #stops training if loss deviance becomes too high
    if gen_ls*1.5<d_ls:
        train_gen = False
        pass
    if d_ls*2<gen_ls:
        train_disc = False
        pass

    if train_disc:
        print('training discriminator')
        session.run(optimizer_disc,feed_dict={noise: n, X_in: batch, is_training:True})

    if train_gen:
        print('training generator')
        session.run(optimizer_gen, feed_dict={noise: n, is_training:True})

    print('iteration %d complete\n'%(iter))

generated_images = session.run(gen, feed_dict={noise: n, is_training:False})
plt.imshow(generated_images[0], interpolation='nearest')
plt.show()
