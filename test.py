import skimage.io as io
import matplotlib.pyplot as plt
from unet import Unet
import sys
import tensorflow as tf
import scipy.misc

name = "testimage/t000.tif"
if len(sys.argv) > 2:
  name = sys.argv[2]

I = io.imread (name)
I = scipy.misc.imresize (I, (572,572))
plt.imshow (I, cmap="Greys")
plt.show()
I = I.reshape ([1,572,572,1])

net = Unet(1)
with tf.Session() as sess:
  sess.run (tf.global_variables_initializer())
  newI = net.get_output (sess, I)
  print (newI.shape)
  plt.imshow (newI[0,:,:,0], cmap="Greys")
  plt.show()
  plt.imshow (newI[0,:,:,1], cmap="Greys")
  plt.show()
