import sys
sys.path.append('../build')
import libPointUtil
import numpy as np
import time
from draw_util import *
from data_util import *
import cv2

def test_downsample():
    pts,_=read_room_h5('data/S3DIS/room/16_Area_1_office_15.h5')

    ds_idxs,_=libPointUtil.gridDownsampleGPU(pts,0.05,True)
    pts=pts[ds_idxs,:]

    spts=np.ascontiguousarray(pts[:,:3])
    nidxs=libPointUtil.findNeighborRadiusGPU(spts,0.2)
    nidxs_lens=np.asarray([len(idxs) for idxs in nidxs],dtype=np.int32)
    nidxs_bgs=compute_nidxs_bgs(nidxs_lens)
    nidxs=np.concatenate(nidxs,axis=0)

    begin=time.time()
    covars=libPointUtil.computeCovarsGPU(spts,nidxs,nidxs_lens,nidxs_bgs)
    print 'cost {} s'.format(time.time()-begin)

    from sklearn.cluster import KMeans
    kmeans=KMeans(5)
    preds=kmeans.fit_predict(covars)
    colors=np.random.randint(0,255,[5,3])
    output_points('test_result/cluster.txt',pts,colors[preds,:])

if __name__=="__main__":
    test_downsample()