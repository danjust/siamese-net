import tensorflow as tf
import numpy as np


def convnet(inputs,reuse=False):

    with tf.variable_scope('conv1',reuse=reuse):
        conv1_1 = tf.layers.conv2d(inputs = inputs,
                                   filters = 64,
                                   kernel_size = 3,
                                   strides = [1,1],
                                   padding = 'SAME',
                                   activation = tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs = conv1_1,
                                    pool_size = 2,
                                    strides = 2,
                                    padding = 'VALID')

    with tf.variable_scope('conv2',reuse=reuse):
        conv2 = tf.layers.conv2d(inputs = pool1,
                                   filters = 128,
                                   kernel_size = 3,
                                   strides = [1,1],
                                   padding = 'SAME',
                                   activation = tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs = conv2,
                                    pool_size = 2,
                                    strides = 2,
                                    padding = 'VALID')

    with tf.variable_scope('conv3',reuse=reuse):
        conv3 = tf.layers.conv2d(inputs = pool2,
                                   filters = 256,
                                   kernel_size = 3,
                                   strides = [1,1],
                                   padding = 'SAME',
                                   activation = tf.nn.relu)

    pool3 = tf.layers.max_pooling2d(inputs = conv3,
                                    pool_size = 2,
                                    strides = 2,
                                    padding = 'VALID')

    with tf.variable_scope('conv4',reuse=reuse):
        conv4 = tf.layers.conv2d(inputs = pool3,
                                   filters = 512,
                                   kernel_size = 3,
                                   strides = [1,1],
                                   padding = 'SAME',
                                   activation = tf.nn.relu)

    pool4 = tf.layers.max_pooling2d(inputs = conv4,
                                    pool_size = 2,
                                    strides = 2,
                                    padding = 'VALID')

    with tf.variable_scope('conv5',reuse=reuse):
        conv5 = tf.layers.conv2d(inputs = pool4,
                                   filters = 512,
                                   kernel_size = 6,
                                   strides = [1,1],
                                   padding = 'VALID',
                                   activation = tf.nn.relu)

    flat = tf.layers.flatten(conv5)

    with tf.variable_scope('fc',reuse=reuse):
        features = tf.layers.dense(inputs=flat, use_bias=False, units=2048, activation=tf.nn.sigmoid)

    return features


def getbatch(imgs, batchsize):
    num_per_char = imgs[0].shape[0]
    diffchar = np.random.randint(0,2,batchsize)
    batch = np.zeros([2,batchsize,105,105,1])
    for i in range(batchsize):
        refchar = np.random.randint(964)
        refiter = np.random.randint(num_per_char)
        if diffchar[i] == 0:
            compchar = refchar
            compiter = np.random.choice(np.setdiff1d(np.arange(num_per_char),[refiter]))
        else:
            compchar = np.random.choice(np.setdiff1d(np.arange(964),[refchar]))
            compiter = np.random.randint(num_per_char)
        batch[0,i,:,:,0] = imgs[refchar][refiter]
        batch[1,i,:,:,0] = imgs[compchar][compiter]
    return batch, np.reshape(diffchar,[-1,1])
