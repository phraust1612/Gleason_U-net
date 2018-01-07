import numpy as np

namelist = [
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

for name in namelist:
  tmp = np.load("../param/W"+name+".npy")
  tmp = tmp.transpose([2,3,1,0])
  np.save ("../param/W"+name+".npy", tmp)
