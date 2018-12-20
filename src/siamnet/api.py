import tensorflow as tf
import numpy as np


class siamnet():
    def __init__(self, img1, img2, target_diff):
        self.img1 = img1
        self.img2 = img2
        self.target_diff = target_diff

        self._results_diff = None
        self._contrastive_loss = None
        self._train_op = None


    @property
    def results_diff(self):
        if self._results_diff is None:
            features1 = convnet(self.img1, reuse=False)
            features2 = convnet(self.img2, reuse=True)
            self._results_diff = tf.sqrt(tf.reduce_mean(tf.square(features1 - features2),
                                                axis=1))
            self._results_diff = tf.reshape(self._results_diff,[-1,1])
        return self._results_diff

    @property
    def contrastive_loss(self, margin=1):
        if self._contrastive_loss is None:
            self._contrastive_loss = (tf.reduce_mean((1-self.target_diff)*self.results_diff**2 / 2
                                      + self.target_diff*(tf.maximum(0.,margin-self.results_diff))**2/2))
        return self._contrastive_loss

    @property
    def train_op(self):
        if self._train_op is None:
            opt = tf.train.AdamOptimizer(
                learning_rate = 0.0001)
            self._train_op = opt.minimize(self.contrastive_loss)
        return self._train_op


    def train(self, imgs_train, trainsteps=5000, printstep=500, batchsize=16):
        l = np.zeros(trainsteps)
        _train_op = self.train_op               # build the model
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(trainsteps):
                batch, batch_diff = getbatch(imgs_train, batchsize)
                _,l[step] = sess.run([_train_op,self.contrastive_loss],
                                     feed_dict={self.img1: batch[0,:,:,:,:],
                                                self.img2: batch[1,:,:,:,:],
                                                self.target_diff: batch_diff})
                if ((step+1)%printstep==0):
                    print('Step %d: Loss %f'
                        % (step,np.mean(l[step-printstep+1:step+1])))

            saver.save(sess, './checkpoints/model', step)


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
