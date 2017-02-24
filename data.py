import numpy as np
import h5py

def batch_generator(batch_size=32):
    data = h5py.File('./data/LowRes_13434_overlapping_pairs.h5','r')
    data = data['dataset_1']
    
    m = batch_size
    while True:
        if m >= data.shape[0]:
            m = batch_size
        batch = data[m-batch_size:m,:,:,:]
        pad1 = np.zeros((batch.shape[0],batch.shape[1],3,batch.shape[3]))
        batch = np.append(batch, pad1,2)
        pad2 = np.zeros((batch.shape[0],2,batch.shape[2],batch.shape[3]))
        batch = np.append(batch, pad2,1)
        
        batch = np.expand_dims(batch,4)
        X = batch[:,:,:,0]
        X /= 255.0
        Y = batch[:,:,:,1]
        Y /= 255.0
        yield (X,Y)
