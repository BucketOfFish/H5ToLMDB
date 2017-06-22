# based on script from https://groups.google.com/forum/#!topic/caffe-users/2xpmLJYmt5k

from numpy import *
import lmdb
import h5py

#import sys
#caffe_root = <where your caffe installation is>
#sys.path.insert(0, caffe_root + 'python');

import caffe

tgt_db = "/home/tn_user/MattWorkspace/ECoGAnalysis/Data/ECoG"
src_db = "/home/tn_user/MattWorkspace/ECoGAnalysis/Data/EC2_blocks_1_8_9_15_76_89_105_CV_HG_align_window_-0.5_to_0.79_file_nobaseline.h5"

env = lmdb.open(tgt_db, map_size=10000000000)

with h5py.File(src_db,'r') as f:
    # extract data from hdf file
    ar_data = array(f['Xhigh gamma'],dtype=float32)
    ar_label = array(f['y'],dtype=int)
    n, h, w = ar_data.shape # n data points, height, width
    c = 1 # channels
    ar_label = ar_label.flatten()
    assert len(ar_label) == n # number of labels has to match the number of input images!
    # write data to lmdb
    for i in range(n):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = c
        datum.height = h
        datum.width = w
        datum.data = ar_data[i,:,:].tobytes()
        datum.label = ar_label[i]
        str_id = '{:08}'.format(i) # create an 8 digit string id based on the index
        with env.begin(write=True) as txn:
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
