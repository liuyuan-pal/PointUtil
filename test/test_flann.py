import sys
sys.path.append('../build')
import libPointUtil
import numpy as np
from draw_util import output_points
import time
from draw_util import *


def test_neighborhood():
    # pt_num=np.random.randint(20000,50000)
    pt_num=10240
    pts=np.random.uniform(0,1,[pt_num,3])
    pts=np.asarray(pts,dtype=np.float32)

    idxs=np.random.choice(pt_num,100,False)
    idxs=np.asarray(idxs,np.int32)
    for i in xrange(10000):
        nidxs=libPointUtil.findNeighborRadiusGPU(pts,idxs,0.1,30)

    # print len(nidxs)
    # for i in range(100):
    #     output_points('test_result/nidxs{}.txt'.format(i),pts[nidxs[i],:])
    #     assert (idxs[i] in list(nidxs[i]))


if __name__=="__main__":
    test_neighborhood()