import sys
sys.path.append('../build')
import libPointUtil
import numpy as np
import time
from draw_util import *
from data_util import *
import os


def test_downsample():
    if not os.path.exists('output'):
        os.mkdir('output')

    pts=uniform_sample_sphere(8192)

    begin=time.time()
    ds_idxs=libPointUtil.gridDownsampleGPU(pts,0.2,False)

    print('cost {} s'.format(time.time()-begin))
    output_points('output/ds_pts.txt',pts[ds_idxs])

    begin=time.time()
    ds_idxs,ds_gidxs=libPointUtil.gridDownsampleGPU(pts,0.2,True)
    print('cost {} s'.format(time.time()-begin))

    colors=np.random.randint(0,256,[len(np.unique(ds_gidxs)),3])
    output_points('output/raw_pts.txt',pts,colors[ds_gidxs])


if __name__=="__main__":
    test_downsample()