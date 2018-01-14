from PIL import Image, ImageDraw
from multiprocessing import Process
import matplotlib.pyplot as plt
import numpy as np
import sys

filter_ = 224
stride_ = 112
NULL_ = 81000000
border_ = 10
Image.MAX_IMAGE_PIXELS = None
class_ = ["3", "4", "5", "n", "i", "s"]

def show_image (I1:np.ndarray, im, x:int, y:int):
  fig = plt.figure ()
  a = fig.add_subplot (1,2,1)
  plt.imshow (I1)
  a.set_title ("Target")
  draw = ImageDraw.Draw (im)
  draw.line ([(x,y),(x+filter_,y)], fill=128, width=border_)
  draw.line ([(x+filter_,y),(x+filter_,y+filter_)], fill=128, width=border_)
  draw.line ([(x+filter_,y+filter_),(x,y+filter_)], fill=128, width=border_)
  draw.line ([(x,y+filter_),(x,y)], fill=128, width=border_)
  del (draw)
  I = np.array(im)
  a = fig.add_subplot (1,2,2)
  plt.imshow (I)
  a.set_title ("Total Image")
  plt.show()


def _main (path):
  anns = []
  img = Image.open(path)
  I = np.array(img)
  x = 0
  while x < I.shape[0]-filter_:
    y = 0
    while y < I.shape[1]-filter_:
      I1 = I[x:x+filter_, y:y+filter_, :]
      print (I1.shape)
      if I1.sum() < NULL_:
        print (x,y, I1.sum())
        p = Process (target=show_image, args=(I1,img,x,y,))
        p.start ()
        ann = input ("enter (3,4,5,n,i,s) : ")
        while ann not in class_:
          ann = input ("try again : ")
        feed = {'x':x, 'y':y, 'annot':ann}
        anns.append (feed)
        p.terminate ()
      y += stride_
    x += stride_
  with open (path+".annot", "w") as f:
    f.write (str (anns))

if __name__ == '__main__':
  for i in sys.argv[1:]:
    _main (i)
