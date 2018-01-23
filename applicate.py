import skimage.io as io
import numpy as np
import tensorflow as tf
import scipy.misc
import openslide
from resnet import Resnet

def readTif (name : str):
  X = openslide.OpenSlide (name)
  X = X.read_region ((0,0),0,(X.dimensions[0],X.dimensions[1]))
  X = np.array (X, dtype='uint8')
  return X

def ratioWSI (I : np.ndarray, batch_size : int):
  annot = classifyWSI (I, batch_size)
  unique, count = np.unique (annot, return_counts=True)
  ans = dict (zip (unique, count))
  del (annot)
  del (unique)
  del (count)
  return ans

def classifyWSI (I : np.ndarray, batch_size : int):
  if I.ndim != 3 or I.shape[2] != 3:
    return -1
  ysize = I.shape[0] // 336
  xsize = I.shape[1] // 336
  ans = np.zeros ([ysize,xsize], dtype='int32')
  net = Resnet()
  with tf.Session() as sess:
    sess.run (tf.global_variables_initializer())
    I1 = np.array ([], dtype='float32')
    I1 = I1.reshape ([0,224,224,3])
    idx = []
    for yi in range (ysize):
      for xi in range (xsize):
        I2 = I[yi*336:(yi+1)*336,xi*336:(xi+1)*336,:]
        I2 = scipy.misc.imresize (I2, (224,224))
        I1 = np.concatenate ([I1, I2])
        del (I2)
        idx.append ((xi,yi))
        if len (idx) == batch_size:
          batch_ans = classifyBatch (I1, net, sess, batch_size)
          for i in range(batch_size):
            ans[idx[i][1]][idx[i][0]] = batch_ans[i]
          del (batch_ans)
          I1 = np.array ([], dtype='float32')
          I1 = I1.reshape ([0,224,224,3])
          idx = []
  del (I1)
  del (idx)
  return ans

def classifyBatch (I : np.ndarray, net : Resnet, sess : tf.Session, batch_size : int):
  output = net.get_output (sess, I)
  ans = np.zeros ([batch_size], dtype='int32')
  for i in range (output.shape[0]):
    ans[i] = np.argmax (output[i])
  del (output)
  return ans

def classifyPatch (I : np.ndarray):
  net = Resnet()
  with tf.Session() as sess:
    sess.run (tf.global_variables_initializer())
    I = I.reshape ([1,224,224,3])
    output = net.get_output (sess, I)
    ans = np.argmax (output[0])
    del (output)
  return ans
