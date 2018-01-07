"""
this modifies the last layer to depth 1
"""
import numpy as np

W = np.load ("../param/W10.npy")
W = W[:,:,:,0]
W = W.reshape ([1,1,64,1])
np.save ("../param/W10.npy", W)
b = np.load ("../param/b10.npy")
b = b[0]
b = b.reshape ([1,])
np.save ("../param/b10.npy", b)
