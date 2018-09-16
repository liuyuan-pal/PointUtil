import h5py
import numpy as np

def read_room_h5(room_h5_file):
    f=h5py.File(room_h5_file,'r')
    data,label = f['data'][:],f['label'][:]
    f.close()

    return data, label


def compute_nidxs_bgs(nidxs_lens):
    csum=0
    nidxs_bgs=np.empty_like(nidxs_lens)
    for i,lval in enumerate(nidxs_lens):
        nidxs_bgs[i]=csum
        csum+=lval
    return nidxs_bgs


# code reference https://zhuanlan.zhihu.com/p/25988652?group_id=828963677192491008
# following code will generate evenly distribution of points on unit sphere
def uniform_sample_sphere(N):
    phi=(np.sqrt(5)-1.0)/2.0
    pts=[]
    for n in range(N):
        z=2.0*n/N-1.0
        x=np.sqrt(1-z**2)*np.cos(2*np.pi*n*phi)
        y=np.sqrt(1-z**2)*np.sin(2*np.pi*n*phi)
        pts.append([x,y,z])

    return np.asarray(pts,np.float32)