import numpy as np
import tensorflow as tf
from gleason import *
from resnet import Resnet

# learning rate decays as scheduled in hlist
epoch=10000 #default epochs
batch_size=10

def main ():
  print ("setting network...")
  net = Resnet()

  total_batch = int(data_len/batch_size)
  last_acc = 0
  with tf.Session () as sess:
    sess.run (tf.global_variables_initializer ())
    print ("start training...")
    try:
      for i in range(epoch):
        avg_loss = 0
        avg_acc = 0
        for j in range(total_batch):
          batch_x, batch_y = gleasonTrainBatch (j*batch_size, (j+1)*batch_size)
          feed = {'x':batch_x,'y':batch_y,'drop':0.5}
          c = net.train_param(sess, feed)
          avg_loss += c/batch_size
        print("epoch:",str(i+1),"loss=",str(avg_loss))

    except KeyboardInterrupt:
      print ("stop learning")
    i = input ("save? [y/n] ")
    while i != 'y' and i != 'n':
      i = input ("enter y or n : ")
    if i == 'y':
      print ("saving weights...")
      net.save (sess)

  print("learning finished")

if __name__ == '__main__':
  print ("start application...")
  main ()
