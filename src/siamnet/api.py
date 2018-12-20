import tensorflow as tf
import numpy as np
import os
from siamnet import convnets


class siamnet():
    def __init__(self, img1, img2, learning_rate,
                 target_diff, model_name,
                 convnet=convnets.VGG16):
        self.img1 = img1
        self.img2 = img2
        self.learning_rate = learning_rate
        self.target_diff = target_diff
        self.model_name = model_name
        self.convnet = convnet

        self._results_diff = None
        self._contrastive_loss = None
        self._train_op = None
        self._summary_op = None

        self.global_step = tf.get_variable('global_step',
                                           initializer=tf.constant(0),
                                           trainable=False)


    @property
    def results_diff(self):
        if self._results_diff is None:
            features1 = self.convnet(self.img1, reuse=False)
            features2 = self.convnet(self.img2, reuse=True)
            self._results_diff = tf.sqrt(tf.reduce_mean(
                tf.square(features1 - features2),
                axis=1))
            self._results_diff = tf.reshape(self._results_diff,[-1,1])
        return self._results_diff

    @property
    def contrastive_loss(self, margin=1):
        if self._contrastive_loss is None:
            self._contrastive_loss = (
                tf.reduce_mean((1-self.target_diff)*self.results_diff**2 / 2
                + self.target_diff*(tf.maximum(0.,margin-self.results_diff))**2/2))
        return self._contrastive_loss

    @property
    def train_op(self):
        if self._train_op is None:
            opt = tf.train.AdamOptimizer(
                learning_rate = self.learning_rate)
            self._train_op = opt.minimize(self.contrastive_loss,
                                          global_step=self.global_step)
        return self._train_op

    @property
    def summary_op(self):
        """Function to write the summary, returns property"""
        if self._summary_op is None:
            tf.summary.scalar("loss", self.contrastive_loss)
            tf.summary.histogram("histogram_loss", self.contrastive_loss)
            self._summary_op =  tf.summary.merge_all()
        return self._summary_op


    def train(self,
              imgs_train,
              trainsteps=5000,
              printstep=500,
              batchsize=16,
              learning_rate=0.0001):
        _train_op = self.train_op               # build the model

        try:
            os.mkdir('./checkpoints/%s' %self.model_name)
        except:
            pass

        saver = tf.train.Saver()
        l = np.zeros(trainsteps)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(
                './checkpoints/%s/checkpoint' % self.model_name))
            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model restored, step = %d" % self.global_step.eval())

            writer_train = tf.summary.FileWriter('./graphs/prediction/train/%s'
                %self.model_name, sess.graph)
            initial_step = self.global_step.eval()

            avg_loss = 0
            for step in range(initial_step, initial_step+trainsteps):
                batch, batch_diff = getbatch(imgs_train, batchsize)
                _,loss,summary = sess.run(
                    [_train_op,self.contrastive_loss,self.summary_op],
                         feed_dict={self.img1: batch[0,:,:,:,:],
                                    self.img2: batch[1,:,:,:,:],
                                    self.target_diff: batch_diff,
                                    self.learning_rate: learning_rate})
                avg_loss = avg_loss+loss/printstep
                if ((step+1)%printstep==0):
                    writer_train.add_summary(summary, global_step=step)
                    print('Step {}: Train loss {:.3f}'.format(step, avg_loss))
                    avg_loss = 0
            writer_train.close()

            saver.save(sess, './checkpoints/%s/training' % self.model_name, step)


def getbatch(imgs, batchsize):
    num_per_char = imgs[0].shape[0]
    diffchar = np.random.randint(0,2,batchsize)
    batch = np.zeros([2,batchsize,105,105,1])
    for i in range(batchsize):
        refchar = np.random.randint(964)
        refiter = np.random.randint(num_per_char)
        if diffchar[i] == 0:
            compchar = refchar
            compiter = np.random.choice(np.setdiff1d(
                np.arange(num_per_char),
                [refiter]))
        else:
            compchar = np.random.choice(np.setdiff1d(np.arange(964),[refchar]))
            compiter = np.random.randint(num_per_char)
        batch[0,i,:,:,0] = imgs[refchar][refiter]
        batch[1,i,:,:,0] = imgs[compchar][compiter]
    return batch, np.reshape(diffchar,[-1,1])
