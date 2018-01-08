import tensorflow as tf
import numpy as np

class Unet:

  def __init__ (self, batch_size):
    """
    __init__ (batch_size):
      initializer of Unet
      you need to decide the batch size to build the network
    """
    self.x = tf.placeholder (tf.float32, [None, 572, 572, 1])
    self.y = tf.placeholder (tf.float32, [None, 3])
    self.tf_drop = tf.placeholder (tf.float32)
    self.W = {}
    self.b = {}
    self.h = 0.025
    self.batch_size = batch_size
    self.namelist = [
      "1_1", "1_2",
      "2_1", "2_2",
      "3_1", "3_2",
      "4_1", "4_2",
      "5_1", "5_2",
      "6_0", "6_1", "6_2",
      "7_0", "7_1", "7_2",
      "8_0", "8_1", "8_2",
      "9_0", "9_1", "9_2",
      "10" ]
    self.load ()
    self.build_net ()

  def load (self):
    """
    load ():
      load weight parameters from param/
    """
    for name in self.namelist:
      nptmp = np.load ("param/W"+name+".npy")
      self.W[name] = tf.Variable (tf.convert_to_tensor(nptmp, name=name))

      nptmp = np.load ("param/b"+name+".npy")
      self.b[name] = tf.Variable (tf.convert_to_tensor(nptmp, name=name))

  def save (self, sess):
    """
    save (sess):
      save weight parameters
      sess : tensorflow session
    """
    for name in self.namelist:
      nptmp = sess.run (self.W[name])
      np.save ("param/W"+name+".npy", nptmp)

      nptmp = sess.run (self.b[name])
      np.save ("param/b"+name+".npy", nptmp)

  def build_net (self):

    """
    TODO : build resnet
    """

  def get_output (self, sess, image):
    """
    get_output (sess, image):
      apply Unet and take the output
      sess : tensorflow session
      image : numpy array of shape : (572, 572, 1)
    """
    _feed = {self.x:image, self.tf_drop:1}
    return sess.run (self.output, feed_dict=_feed)

  def train_param (self, sess, feed):
    """
    train_param (sess, feed):
      train and return loss
      sess : tensorflow session
      feed : dict {'x', 'y', 'drop'}
    """
    _feed = {self.x:feed['x'], self.y:feed['y'], self.tf_drop:feed['drop']}
    c,_ = sess.run([self.loss, self.train], feed_dict=_feed)
    return c
