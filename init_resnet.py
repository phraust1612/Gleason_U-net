import numpy as np
import os

d = 'param_resnet/'
l = os.listdir(d)
for i in l:
  x = np.load (d+i)
  if x.ndim == 4:
    x = x.transpose (2,3,1,0)
    np.save (d+i, x)

x = np.random.randn(2048,6)
x = x.astype ('float32')
np.save (d+"fc6_0.npy", x)
x = np.random.randn(6)
x = x.astype ('float32')
np.save (d+"fc6_1.npy", x)
