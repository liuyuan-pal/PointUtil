import sys
sys.path.append('../build')
import libPointUtil
import numpy as np
import time
from draw_util import *
from data_util import *


def test_downsample():
    pts,_=read_room_h5('data/S3DIS/room/38_Area_1_office_6.h5')

    begin=time.time()
    ds_idxs=libPointUtil.gridDownsampleGPU(pts,0.05,False)
    print 'cost {} s'.format(time.time()-begin)

    begin=time.time()
    for i in xrange(100):
        ds_idxs,ds_gidxs=libPointUtil.gridDownsampleGPU(pts,0.05,True)
    print 'cost {} s'.format((time.time()-begin)/100.0)

    print ds_idxs.shape
    output_points('test_result/ds.txt',pts[ds_idxs,:])
    ds_num=ds_idxs.shape[0]
    gcolors=np.random.randint(0,256,[ds_num,3])
    output_points('test_result/ds_gidxs.txt',pts,gcolors[ds_gidxs,:])

if __name__=="__main__":
    test_downsample()