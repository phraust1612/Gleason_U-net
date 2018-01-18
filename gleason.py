import skimage.io as io
import scipy.misc
import numpy as np
import os

train_dir = "training_set/"
anns = os.listdir (train_dir)
max_index = []
for i in anns:
  flist = os.listdir (train_dir + i)
  max_index.append (len (flist) + flist[-1])

data_len = max_index[-1]

def gleasonTrainBatch (i:int, j:int):
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
  flist = os.listdir (train_dir + anns[start_label_idx])
  start_idx = i
  end_idx = j
  while i >= max_index[start_label_idx]:
    start_idx -= len (flist)
    end_idx -= len (flist)
    start_label_idx += 1
    flist = os.listdir (train_dir + anns[start_label_idx])

  last_label_idx = start_label_idx
  while j > max_index[last_label_idx]:
    end_idx -= len (flist)
    last_label_idx += 1
    flist = os.listdir (train_dir + anns[last_label_idx])

  for label_idx in range (start_label_idx, last_label_idx+1):
    flist = os.listdir (train_dir + anns[label_idx])
    for idx in range (start_idx, len (flist)):
      if idx == end_idx and label_idx == last_label_idx:
        break
      tmp = io.imread (train_dir + flist[start_label_idx] + "/" + flist[fi])
      tmp = ImageProcess (tmp)
      data = np.concatenate ([data, tmp])
      tmp2 = np.eye (len (anns), dtype='int32')[label_idx]
      tmp2.transpose()
      label = np.concatenate ([label, tmp2])
    start_idx = 0

  del (tmp)
  return data, label

def ImageProcess (I:np.ndarray):
  I = scipy.misc.imresize (I, (224,224))
  I.astype ("float32")
  I -= np.mean (I)
  I /= np.std (I)
  return I
