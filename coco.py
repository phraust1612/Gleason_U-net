import sys
import numpy as np
import scipy.misc
import tensorflow as tf
import skimage.io as io
from pycocotools.coco import COCO
from PIL import Image,ImageDraw
from unet import Unet
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

net = Unet(batch_size)
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
  sess.run(tf.global_variables_initializer())

  total_batch = int(len(DATA)/batch_size)
  try:
    for i in range(epoch):
      avg_loss = 0
      for j in range(total_batch):
        batch_x = np.array(DATA[j*batch_size:(j+1)*batch_size])
        batch_y = np.array(ANNOT[j*batch_size:(j+1)*batch_size])
        feed_dic= {net.x:batch_x,net.y:batch_y,net.tf_drop:drop_rate}
        c,_ = sess.run([net.loss, net.train],feed_dict=feed_dic)
        avg_loss += c/batch_size
      print("epoch:",str(i+1),"loss=",str(avg_loss))
  except KeyboardInterrupt:
    print("learning finished")
