import sys
sys.path.append('../build')
import libPointUtil
import numpy as np
import time
from draw_util import *
from data_util import *
import random
import math

def test_downsample():
    pts,_=read_room_h5('data/S3DIS/room/45_Area_2_auditorium_2.h5')

    ds_idxs,ds_gidxs=libPointUtil.gridDownsampleGPU(pts,0.05,True)

    pts=pts[ds_idxs,:]
    angle=random.random()*math.pi/2.0
    begin=time.time()
    block_idxs=libPointUtil.sampleRotatedBlockGPU(pts,1.5,3.0,angle)
    print 'cost {} s'.format(time.time()-begin)
    for i,idxs in enumerate(block_idxs):
        if idxs.shape[0]==0:
            continue
        print idxs.shape
        output_points('test_result/block{}.txt'.format(i),pts[idxs])

if __name__=="__main__":
    test_downsample()