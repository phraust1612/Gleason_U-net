import skimage.io as io
import scipy.misc
import numpy as np
import os
import stainNorm_Reinhard as reinhard

class Gleason:
  def __init__ (self, dir_name):
    self.train_dir = dir_name
    self.anns = os.listdir (self.train_dir)
    self.max_index = []
    for i in self.anns:
      flist = os.listdir (self.train_dir + i)
      if len(self.max_index) > 0:
        self.max_index.append (len (flist) + self.max_index[-1])
      else:
        self.max_index.append (len (flist))

    self.data_len = self.max_index[-1]
    self.ref_img=np.array(scipy.misc.imresize(io.imread("color_norm_ref.jpg"),(224,224)), dtype="float32")

  def gleasonBatch (self, i:int, j:int):
    data = np.array ([], dtype=np.float32)
    data = data.reshape ([0,224,224,3])
    label = np.array ([], dtype=np.int32)
    label = data.reshape ([0,6])

    """
    ex>
    [dir0] [dir1] ... [dir5]
      |                |
      v                v
    123th      ~     234th
  
    start_label_idx = 0
    last_label_idx = 5
    start_idx = 123
    end_idx = 235
    """
    start_label_idx = 0
    flist = os.listdir (self.train_dir + self.anns[start_label_idx])
    start_idx = i
    end_idx = j
    while i >= self.max_index[start_label_idx]:
      start_idx -= len (flist)
      end_idx -= len (flist)
      start_label_idx += 1
      flist = os.listdir (self.train_dir + self.anns[start_label_idx])

    last_label_idx = start_label_idx
    while j > self.max_index[last_label_idx]:
      end_idx -= len (flist)
      last_label_idx += 1
      flist = os.listdir (self.train_dir + self.anns[last_label_idx])

    for label_idx in range (start_label_idx, last_label_idx+1):
      flist = os.listdir (self.train_dir + self.anns[label_idx])
      for idx in range (start_idx, len (flist)):
        if idx == end_idx and label_idx == last_label_idx:
          break
        tmp = io.imread (self.train_dir + self.anns[label_idx] + "/" + flist[idx])
        tmp = self.ImageProcess (tmp)
        tmp = tmp.reshape (1,224,224,3)
        data = np.concatenate ([data, tmp])
        tmp2 = np.eye (len (self.anns), dtype='int32')[label_idx]
        tmp2.transpose()
        tmp2 = tmp2.reshape (1,6)
        label = np.concatenate ([label, tmp2])
      start_idx = 0

    del (tmp)
    del (tmp2)
    del (flist)
    return data, label

  def ImageProcess (self, img:np.ndarray):
    I = scipy.misc.imresize (img, (224,224))
    I = np.array (I, dtype="float32")
    n=reinhard.normalizer()
    n.fit(self.ref_img)
    I=n.transform(I)
    del(n)
    return I
