import sys
import numpy as np
import scipy.misc
import tensorflow as tf
import skimage.io as io
from pycocotools.coco import COCO
from PIL import Image,ImageDraw
# import matplotlib.pyplot as plt

DATA = []
ANNOT = []

## hyperparams
# learning rate decays as scheduled in hlist
hlist=[0.0001, 0.0125]
h=0.025
epoch=100
batch_size=1
drop_rate = 0.5

## load images and their annotation map
## you may use other datasets
# COCO dataset
# img : dict_keys(['height', 'coco_url', 'width', 'file_name', 'flickr_url', 'date_captured', 'license', 'id'])
# I : (height, width, depth)
# anns[j] : j'th object - dict_keys(['iscrowd', 'area', 'segmentation', 'bbox', 'image_id', 'id', 'category_id'])
coco = COCO("coco/annotations/instances_val2017.json")
catIds = coco.getCatIds(catNms=['person','dog','skateboard'])
imgIds = coco.getImgIds(catIds=catIds)
print("loading datasets...")
for i in range (len (imgIds)):
  img = coco.loadImgs(imgIds[i])[0]
  I = io.imread(img['coco_url'])
  I = scipy.misc.imresize(I, (572,572))
  DATA.append (I)
  annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
  anns = coco.loadAnns(annIds)
  mimg = Image.new('L', (img['width'], img['height']), 0)
  for j in range (len (anns)):
    ImageDraw.Draw(mimg).polygon(anns[j]['segmentation'][0], outline=anns[j]['category_id'], fill=anns[j]['category_id'])
  mask = np.array(mimg)
  mask = scipy.misc.imresize(mask, (388,388))
  ANNOT.append (mask)
  ## uncommnet below to test annotation map images
  # plt.imshow(mask)
  # plt.show()
  print ("%02.2f"% (100 * (i+1)/len(imgIds)) + "% done")
print("loading done...")

x=tf.placeholder(tf.float32,[None,572,572,3])
y=tf.placeholder(tf.float32,[None,388,388])
tf_drop = tf.placeholder(tf.float32)

L1_0 = tf.reshape(x,[-1,572,572,3])
# no need to transpose - see manual of value Tensor shape
# L1_0 = tf.transpose(L1_0,perm=[0,2,1,3])

# L1_0 : 572 x 572 x 3
# 3x3 conv + ReLU layer
# weight initial state : gaussian distribution with standard deviation 0.5
W1_1 = tf.Variable(tf.random_normal([3,3,3,64],stddev=0.5),name="W1_1")
L1_1 = tf.nn.conv2d(L1_0,W1_1,strides=[1,1,1,1],padding="VALID")
L1_1 = tf.nn.relu(L1_1)

# 3x3 conv + ReLU layer
W1_2 = tf.Variable(tf.random_normal([3,3,64,64],stddev=0.5),name="W1_2")
L1_2 = tf.nn.conv2d(L1_1,W1_2,strides=[1,1,1,1],padding="VALID")
L1_2 = tf.nn.relu(L1_2)

# 2x2 max-pooling layer
L2_0 = tf.nn.max_pool(L1_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
L2_0 = tf.nn.dropout(L2_0,keep_prob=tf_drop)

# L2_0 : 284 x 284 x 64
# 3x3 conv + ReLU layer
W2_1 = tf.Variable(tf.random_normal([3,3,64,128],stddev=0.5),name="W2_1")
L2_1 = tf.nn.conv2d(L2_0,W2_1,strides=[1,1,1,1],padding="VALID")
L2_1 = tf.nn.relu(L2_1)

# 3x3 conv + ReLU layer
W2_2 = tf.Variable(tf.random_normal([3,3,128,128],stddev=0.5),name="W2_2")
L2_2 = tf.nn.conv2d(L2_1,W2_2,strides=[1,1,1,1],padding="VALID")
L2_2 = tf.nn.relu(L2_2)

# 2x2 max-pooling layer
L3_0 = tf.nn.max_pool(L2_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
L3_0 = tf.nn.dropout(L3_0,keep_prob=tf_drop)

# L3_0 : 140 x 140 x 128
# 3x3 conv + ReLU layer
W3_1 = tf.Variable(tf.random_normal([3,3,128,256],stddev=0.5),name="W3_1")
L3_1 = tf.nn.conv2d(L3_0,W3_1,strides=[1,1,1,1],padding="VALID")
L3_1 = tf.nn.relu(L3_1)

# 3x3 conv + ReLU layer
W3_2 = tf.Variable(tf.random_normal([3,3,256,256],stddev=0.5),name="W3_2")
L3_2 = tf.nn.conv2d(L3_1,W3_2,strides=[1,1,1,1],padding="VALID")
L3_2 = tf.nn.relu(L3_2)

# 2x2 max-pooling layer
L4_0 = tf.nn.max_pool(L3_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
L4_0 = tf.nn.dropout(L4_0,keep_prob=tf_drop)

# L4_0 : 68 x 68 x 256
# 3x3 conv + ReLU layer
W4_1 = tf.Variable(tf.random_normal([3,3,256,512],stddev=0.5),name="W4_1")
L4_1 = tf.nn.conv2d(L4_0,W4_1,strides=[1,1,1,1],padding="VALID")
L4_1 = tf.nn.relu(L4_1)

# 3x3 conv + ReLU layer
W4_2 = tf.Variable(tf.random_normal([3,3,512,512],stddev=0.5),name="W4_2")
L4_2 = tf.nn.conv2d(L4_1,W4_2,strides=[1,1,1,1],padding="VALID")
L4_2 = tf.nn.relu(L4_2)

# 2x2 max-pooling layer
L5_0 = tf.nn.max_pool(L4_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
L5_0 = tf.nn.dropout(L5_0,keep_prob=tf_drop)

# L5_0 : 32 x 32 x 512
# 3x3 conv + ReLU layer
W5_1 = tf.Variable(tf.random_normal([3,3,512,1024],stddev=0.5),name="W5_1")
L5_1 = tf.nn.conv2d(L5_0,W5_1,strides=[1,1,1,1],padding="VALID")
L5_1 = tf.nn.relu(L5_1)

# 3x3 conv + ReLU layer
W5_2 = tf.Variable(tf.random_normal([3,3,1024,1024],stddev=0.5),name="W5_2")
L5_2 = tf.nn.conv2d(L5_1,W5_2,strides=[1,1,1,1],padding="VALID")
L5_2 = tf.nn.relu(L5_2)

# 2x2 up-conv + concatenation
W6_0 = tf.Variable(tf.random_normal([2,2,512,1024],stddev=0.5),name="W6_0")
L6_0 = tf.nn.conv2d_transpose(L5_2,W6_0,[batch_size,56,56,512],strides=[1,2,2,1],padding="VALID")
L4_3 = tf.image.resize_image_with_crop_or_pad(L4_2, 56, 56)
L6_0 = tf.concat([L4_3,L6_0],-1)

# L6_0 : 56 x 56 x 1024
# 3x3 conv + ReLU layer
W6_1 = tf.Variable(tf.random_normal([3,3,1024,512],stddev=0.5),name="W6_1")
L6_1 = tf.nn.conv2d(L6_0,W6_1,strides=[1,1,1,1],padding="VALID")
L6_1 = tf.nn.relu(L6_1)

# 3x3 conv + ReLU layer
W6_2 = tf.Variable(tf.random_normal([3,3,512,512],stddev=0.5),name="W6_2")
L6_2 = tf.nn.conv2d(L6_1,W6_2,strides=[1,1,1,1],padding="VALID")
L6_2 = tf.nn.relu(L6_2)

# 2x2 up-conv + concatenation
W7_0 = tf.Variable(tf.random_normal([2,2,256,512],stddev=0.5),name="W7_0")
L7_0 = tf.nn.conv2d_transpose(L6_2,W7_0,[batch_size,104,104,256],strides=[1,2,2,1],padding="VALID")
L3_3 = tf.image.resize_image_with_crop_or_pad(L3_2, 104, 104)
L7_0 = tf.concat([L3_3,L7_0],-1)

# L7_0 : 104 x 104 x 512
# 3x3 conv + ReLU layer
W7_1 = tf.Variable(tf.random_normal([3,3,512,256],stddev=0.5),name="W7_1")
L7_1 = tf.nn.conv2d(L7_0,W7_1,strides=[1,1,1,1],padding="VALID")
L7_1 = tf.nn.relu(L7_1)

# 3x3 conv + ReLU layer
W7_2 = tf.Variable(tf.random_normal([3,3,256,256],stddev=0.5),name="W7_2")
L7_2 = tf.nn.conv2d(L7_1,W7_2,strides=[1,1,1,1],padding="VALID")
L7_2 = tf.nn.relu(L7_2)

# 2x2 up-conv + concatenation
W8_0 = tf.Variable(tf.random_normal([2,2,128,256],stddev=0.5),name="W8_0")
L8_0 = tf.nn.conv2d_transpose(L7_2,W8_0,[batch_size,200,200,128],strides=[1,2,2,1],padding="VALID")
L2_3 = tf.image.resize_image_with_crop_or_pad(L2_2, 200, 200)
L8_0 = tf.concat([L2_3,L8_0],-1)

# L8_0 : 200 x 200 x 256
# 3x3 conv + ReLU layer
W8_1 = tf.Variable(tf.random_normal([3,3,256,128],stddev=0.5),name="W8_1")
L8_1 = tf.nn.conv2d(L8_0,W8_1,strides=[1,1,1,1],padding="VALID")
L8_1 = tf.nn.relu(L8_1)

# 3x3 conv + ReLU layer
W8_2 = tf.Variable(tf.random_normal([3,3,128,128],stddev=0.5),name="W8_2")
L8_2 = tf.nn.conv2d(L8_1,W8_2,strides=[1,1,1,1],padding="VALID")
L8_2 = tf.nn.relu(L8_2)

# 2x2 up-conv + concatenation
W9_0 = tf.Variable(tf.random_normal([2,2,64,128],stddev=0.5),name="W9_0")
L9_0 = tf.nn.conv2d_transpose(L8_2,W9_0,[batch_size,392,392,64],strides=[1,2,2,1],padding="VALID")
L1_3 = tf.image.resize_image_with_crop_or_pad(L1_2, 392, 392)
L9_0 = tf.concat([L1_3,L9_0],-1)

# L9_0 : 392 x 392 x 128
# 3x3 conv + ReLU layer
W9_1 = tf.Variable(tf.random_normal([3,3,128,64],stddev=0.5),name="W9_1")
L9_1 = tf.nn.conv2d(L9_0,W9_1,strides=[1,1,1,1],padding="VALID")
L9_1 = tf.nn.relu(L9_1)

# 3x3 conv + ReLU layer
W9_2 = tf.Variable(tf.random_normal([3,3,64,64],stddev=0.5),name="W9_2")
L9_2 = tf.nn.conv2d(L9_1,W9_2,strides=[1,1,1,1],padding="VALID")
L9_2 = tf.nn.relu(L9_2)

# 1x1 conv
W10 = tf.Variable(tf.random_normal([1,1,64,1],stddev=0.5),name="W10")
L10 = tf.nn.conv2d(L9_2,W10,strides=[1,1,1,1],padding="SAME")

# print ("L10 : "+ str (L10.shape))
# final layer L10 : 388 x 388 x 1

# L10 = tf.transpose(L10,perm=[0,2,1,3])
L10 = tf.reshape(L10, [-1,388,388])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=L10,labels=y))
optimizer = tf.train.AdagradOptimizer(learning_rate=h).minimize(loss)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
  sess.run(tf.global_variables_initializer())

  total_batch = int(len(DATA)/batch_size)
  try:
    for i in range(epoch):
      avg_loss = 0
      for j in range(total_batch):
        batch_x = np.array(DATA[j*batch_size:(j+1)*batch_size])
        batch_y = np.array(ANNOT[j*batch_size:(j+1)*batch_size])
        feed_dic= {x:batch_x,y:batch_y,tf_drop:drop_rate}
        c,_ = sess.run([loss,optimizer],feed_dict=feed_dic)
        avg_loss += c/batch_size
      print("epoch:",str(i+1),"loss=",str(avg_loss),"h=",h)
  except KeyboardInterrupt:
    print("learning finished")
