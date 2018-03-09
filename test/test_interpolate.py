import numpy as np
import sys
sys.path.append('../build')
import libPointUtil
from data_util import compute_nidxs_bgs
from draw_util import output_points

def test_interpolate():
    sxyzs=np.asarray([[0.0,0.0,0.0],
                    [0.0,1.0,0.0],
                    [1.0,0.0,0.0],
                    [1.0,1.0,0.0],
                    ],dtype=np.float32)
    # sxyzs=np.random.uniform(0.0,1.0,[128,3])
    sxyzs=np.asarray(sxyzs,dtype=np.float32)
    sprobs=np.random.uniform(0.0,255.0,[4,3])
    print sprobs
    sprobs=np.asarray(sprobs,dtype=np.float32)

    qxyzs=np.random.uniform(0.0,1.0,[1024,3])
    qxyzs[:,2]=0.0
    qxyzs=np.asarray(qxyzs,dtype=np.float32)

    nidxs=libPointUtil.findNeighborInAnotherGPU(sxyzs,qxyzs,4,1)
    nidxs_lens=np.asarray([len(idxs) for idxs in nidxs],dtype=np.int32)
    nidxs_bgs=compute_nidxs_bgs(nidxs_lens)
    nidxs=np.concatenate(nidxs,axis=0)

    qprobs=libPointUtil.interpolateProbsGPU(sxyzs,qxyzs,sprobs,nidxs,nidxs_lens,nidxs_bgs,2)
    print qprobs
    output_points('test_result/int_rgbs.txt',qxyzs,np.asarray(qprobs,np.int))

if __name__=="__main__":
    test_interpolate()

