import numpy as np
import h5py

f = h5py.File('getting started toy datasets/test_signs.h5', 'r')
dset = f['key']
data = np.array(dset[:,:,:])
print(data.shape)
file = 'test.jpg'
#cv2.imwrite(file, data)