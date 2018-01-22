import tensorflow as tf
import numpy as np
import os

class Resnet:

  def __init__ (self):
    """
    __init__ ():
      initializer of Resnet-152
    """
    self.x = tf.placeholder (tf.float32, [None, 224, 224, 3])
    self.y = tf.placeholder (tf.float32, [None, 6])
    self.tf_drop = tf.placeholder (tf.float32)
    self.W = {}
    self.h = 0.00001
    self.classifier = "svm"
    self.param_dir = "param_resnet/"
    self.namelist = os.listdir(self.param_dir)
    self.load ()
    self.build_net ()

  def load (self):
    """
    load ():
      load weight parameters from param/
    """
    for name in self.namelist:
      nptmp = np.load (self.param_dir+name)
      self.W[name] = tf.Variable (tf.convert_to_tensor(nptmp, name=name))

  def save (self, sess):
    """
    save (sess):
      save weight parameters
      <arguments>
        sess : tensorflow session
    """
    for name in self.namelist:
      nptmp = sess.run (self.W[name])
      np.save (self.param_dir+name, nptmp)

  def res_cycle (self, x, level, branch, level2, padding, relu, strides=1):
    """
    res_cycle (x, level, branch, level2, padding, relu):
      a cycle which computes conv - batch norm - scale - relu (optional)
      <arguments>
        x : input tensor
        level : string, first level (2a, 2b, etc)
        branch : int or string - should be 1 or 2
        level2 : string, second level which might differ in case of second branch
        padding : SAME or VALID
        relu : boolean
    """
    if type (level) != str:
      level = str(level)
    if type (branch) != str:
      branch = str (branch)

    L = tf.nn.conv2d (x, self.W["res"+level+"_branch"+branch+level2+"_0.npy"], strides=[1,strides,strides,1], padding=padding)
    L = tf.nn.batch_normalization (L,
      self.W['bn'+level+"_branch"+branch+level2+'_0.npy'],
      self.W['bn'+level+"_branch"+branch+level2+'_1.npy'],
      self.W['scale'+level+"_branch"+branch+level2+'_1.npy'],
      self.W['scale'+level+"_branch"+branch+level2+'_0.npy'],
      0.00001)
    if relu:
      L = tf.nn.relu (L)
    return L

  def secondary_cycle (self, x, level, strides=1):
    """
    secondary_cycle (x, level):
      a cycle of secondary branch
      which calls res_cycle three times
      <arguments>
        x : input tensor
        level : string, first level
    """
    if type (level) != str:
      level = str(level)

    L = self.res_cycle (x, level, 2, 'a', "VALID", True, strides)
    L = self.res_cycle (L, level, 2, 'b', "SAME", True)
    L = self.res_cycle (L, level, 2, 'c', "VALID", False)
    return L

  def primary_cycle (self, L, level_scope, strides=1):
    """
    primary_cycle (L, level_scope):
      a cycle of the highest level
      <arguments>
        L : input tensor
        level_scope : list of levels to run iteratively
        strides : strides for the first layer
    """
    for i in level_scope:
      # branch 2
      L2 = self.secondary_cycle (L, i, strides)

      # branch 1
      if level_scope.index (i) == 0:
        L = self.res_cycle (L, i, 1, '', "VALID", False, strides)

      strides = 1
      L = L + L2
      L = tf.nn.relu (L)
    return L

  def build_net (self):
    """
    build_net ():
      build renet architecture
    """
    L = tf.reshape (self.x, [-1, 224, 224, 3])

    # res1
    L = tf.pad (L, [[0,0],[3,3],[3,3],[0,0]], "CONSTANT")
    L = tf.nn.conv2d (L, self.W["conv1_0.npy"], strides=[1,2,2,1], padding="VALID")
    L = tf.nn.batch_normalization (L,
      self.W['bn_conv1_0.npy'],
      self.W['bn_conv1_1.npy'], 
      self.W['scale_conv1_1.npy'],
      self.W['scale_conv1_0.npy'],
      0.00001)
    L = tf.nn.relu (L)
    L = tf.nn.max_pool (L, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME")

    # res2
    l = ['2a','2b','2c']
    L = self.primary_cycle (L, l)

    # res3
    l = ['3a']
    for i in range(1, 8):
      l.append ('3b'+str(i))
    L = self.primary_cycle (L, l, 2)

    # res4
    l = ['4a']
    for i in range(1, 36):
      l.append ('4b'+str(i))
    L = self.primary_cycle (L, l, 2)

    # res5
    l = ['5a','5b','5c']
    L = self.primary_cycle (L, l, 2)

    L = tf.nn.avg_pool (L, ksize=[1,7,7,1], strides=[1,1,1,1], padding="VALID")
    L = tf.reshape (L, [-1, 2048])
    self.output = tf.matmul (L, self.W['fc6_0.npy']) + self.W['fc6_1.npy']

    # you may use SVM or softmax classifier here
    if self.classifier == "softmax":
        self.loss = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits
            (logits=self.output, labels=self.y))
    else:
        self.loss = tf.losses.hinge_loss (self.y, self.output)
    optimizer = tf.train.AdagradOptimizer (learning_rate = self.h)
    self.train = optimizer.minimize (self.loss)

    correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  def get_output (self, sess, image):
    """
    get_output (sess, image):
      apply resnet and take the score function
      <arguments>
        sess : tensorflow session
        image : numpy array of shape : (224, 224, 1)
    """
    _feed = {self.x:image, self.tf_drop:1}
    return sess.run (self.output, feed_dict=_feed)

  def get_accuracy (self, sess, feed):
    """
    get_output (sess, image):
      test resnet and take the accuracy
      <arguments>
        sess : tensorflow session
        feed : dict {'x', 'y'}
    """
    _feed = {self.x:feed['x'], self.y:feed['y'], self.tf_drop:1}
    return sess.run (self.accuracy, feed_dict=_feed)

  def train_param (self, sess, feed):
    """
    train_param (sess, feed):
      train and return loss
      <arguments>
        sess : tensorflow session
        feed : dict {'x', 'y', 'drop'}
    """
    _feed = {self.x:feed['x'], self.y:feed['y'], self.tf_drop:feed['drop']}
    c,_ = sess.run([self.loss, self.train], feed_dict=_feed)
    return c
