import sys
sys.path.append('../build')
import libPointUtil
import numpy as np
import time
from draw_util import *
from data_util import *

def test_covars():
    pts,_=read_room_h5('data/S3DIS/room/16_Area_1_office_15.h5')

    ds_idxs,_=libPointUtil.gridDownsampleGPU(pts,0.05,True)
    pts=pts[ds_idxs,:]

    spts=np.ascontiguousarray(pts[:,:3])
    nidxs=libPointUtil.findNeighborRadiusGPU(spts,0.2)
    nidxs_lens=np.asarray([len(idxs) for idxs in nidxs],dtype=np.int32)
    nidxs_bgs=compute_nidxs_bgs(nidxs_lens)
    nidxs=np.concatenate(nidxs,axis=0)

    begin=time.time()
    for i in xrange(10000):
        covars=libPointUtil.computeCovarsGPU(spts,nidxs,nidxs_lens,nidxs_bgs)
    print 'cost {} s'.format(time.time()-begin)

    from sklearn.cluster import KMeans
    kmeans=KMeans(5)
    preds=kmeans.fit_predict(covars)
    colors=np.random.randint(0,255,[5,3])
    output_points('test_result/cluster.txt',pts,colors[preds,:])


def test_voxel_sort():
    pts,_=read_room_h5('data/S3DIS/room/16_Area_1_office_15.h5')

    ds_idxs,_=libPointUtil.gridDownsampleGPU(pts,0.05,True)
    cxyz1=pts[ds_idxs,:]

    cxyz1=np.ascontiguousarray(cxyz1)
    sidxs1,vlens1=libPointUtil.sortVoxelGPU(cxyz1,0.15)

    cxyz1=cxyz1[sidxs1,:]
    cxyz1=np.ascontiguousarray(cxyz1)
    dxyz1,cxyz2=libPointUtil.computeCenterDiffCPU(cxyz1,vlens1)

    # output
    vidxs=[]
    for i,l in enumerate(vlens1):
        vidxs+=[i for _ in xrange(l)]
    colors=np.random.randint(0,256,[vlens1.shape[0],3])
    vidxs=np.asarray(vidxs,np.int32)

    output_points('test_result/cxyz1t.txt',cxyz1,colors[vidxs,:])
    output_points('test_result/cxyz2t.txt',cxyz2,colors)

    # sort v2
    sidxs2,vlens2=libPointUtil.sortVoxelGPU(cxyz2,0.5)
    cxyz2=cxyz2[sidxs2,:]
    cxyz2=np.ascontiguousarray(cxyz2)
    dxyz2,cxyz3=libPointUtil.computeCenterDiffCPU(cxyz2,vlens2)

    print sidxs1.shape
    print cxyz1.shape[0]
    sidxs1,vlens1=libPointUtil.adjustPointsMemoryCPU(vlens1,sidxs2,cxyz1.shape[0])

    print sidxs1.shape
    cxyz1=cxyz1[sidxs1,:]
    vidxs=[]
    for i,l in enumerate(vlens1):
        vidxs+=[i for _ in xrange(l)]
    colors=np.random.randint(0,256,[vlens1.shape[0],3])
    vidxs=np.asarray(vidxs,np.int32)

    output_points('test_result/cxyz1.txt',cxyz1,colors[vidxs,:])
    output_points('test_result/cxyz2.txt',cxyz2,colors)

    vidxs=[]
    for i,l in enumerate(vlens2):
        vidxs+=[i for _ in xrange(l)]
    colors=np.random.randint(0,256,[vlens2.shape[0],3])
    vidxs=np.asarray(vidxs,np.int32)

    output_points('test_result/cxyz2a.txt',cxyz2,colors[vidxs,:])
    output_points('test_result/cxyz3a.txt',cxyz3,colors)


def permutation(feats_list,idxs):
    for i in xrange(len(feats_list)):
        feats_list[i]=feats_list[i][idxs]
    return feats_list


def build_hierarchy(xyzs,feats_list,vs1,vs2):

    ###########################

    cxyz1=np.ascontiguousarray(xyzs)
    sidxs1,vlens1=libPointUtil.sortVoxelGPU(cxyz1,vs1)

    cxyz1=cxyz1[sidxs1,:]
    cxyz1=np.ascontiguousarray(cxyz1)
    dxyz1,cxyz2=libPointUtil.computeCenterDiffCPU(cxyz1,vlens1)

    feats_list=permutation(feats_list,sidxs1)

    ############################

    cxyz2=np.ascontiguousarray(cxyz2)
    sidxs2,vlens2=libPointUtil.sortVoxelGPU(cxyz2,vs2)

    cxyz2=cxyz2[sidxs2,:]
    cxyz2=np.ascontiguousarray(cxyz2)
    dxyz2,cxyz3=libPointUtil.computeCenterDiffCPU(cxyz2,vlens2)

    sidxs1,vlens1=libPointUtil.adjustPointsMemoryCPU(vlens1,sidxs2,cxyz1.shape[0])
    dxyz1,cxyz1=permutation([dxyz1,cxyz1],sidxs1)
    feats_list=permutation(feats_list,sidxs1)

    return cxyz1,dxyz1,vlens1,cxyz2,dxyz2,vlens2,cxyz3,feats_list


def test_hierachy():
    pts,lbls=read_room_h5('data/S3DIS/room/12_Area_1_office_11.h5')
    ds_idxs=libPointUtil.gridDownsampleGPU(pts,0.05,False)
    pts=pts[ds_idxs,:]
    lbls=lbls[ds_idxs]
    xyzs=pts[:,:3]
    rgbs=pts[:,3:]
    cxyz1, dxyz1, vlens1, cxyz2, dxyz2, vlens2, cxyz3, feats_list = build_hierarchy(xyzs,[rgbs,lbls],0.15,0.5)
    rgbs, lbls = feats_list

    output_points('test_result/cxyz1_rgb.txt',cxyz1,rgbs)
    colors=get_class_colors()
    output_points('test_result/cxyz1_lbl.txt',cxyz1,colors[lbls.flatten(),:])

    # test cxyz
    vidxs=[]
    for i,l in enumerate(vlens1):
        vidxs+=[i for _ in xrange(l)]
    colors=np.random.randint(0,256,[vlens1.shape[0],3])
    vidxs=np.asarray(vidxs,np.int32)

    output_points('test_result/cxyz1.txt',cxyz1,colors[vidxs,:])
    output_points('test_result/cxyz2.txt',cxyz2,colors)

    vidxs=[]
    for i,l in enumerate(vlens2):
        vidxs+=[i for _ in xrange(l)]
    colors=np.random.randint(0,256,[vlens2.shape[0],3])
    vidxs=np.asarray(vidxs,np.int32)

    output_points('test_result/cxyz2a.txt',cxyz2,colors[vidxs,:])
    output_points('test_result/cxyz3a.txt',cxyz3,colors)

    # test dxyz
    c=0
    for k,l in enumerate(vlens1):
        for t in xrange(l):
            dxyz1[c+t]+=cxyz2[k]
        c+=l
    output_points('test_result/dxyz1.txt',dxyz1)

    c=0
    for k,l in enumerate(vlens2):
        for t in xrange(l):
            dxyz2[c+t]+=cxyz3[k]
        c+=l
    output_points('test_result/dxyz2.txt',dxyz2)




if __name__=="__main__":
    test_hierachy()