#include "CudaUtil.h"

#include <cmath>

__global__
void computeCovarsGPUKernel(
        float *pts,             // [pn,ps]
        int *nidxs,             // [csum]
        int *nidxs_lens,        // [pn]
        int *nidxs_bgs,         // [pn]
        float *covars,          // [pn,9]
        int pn,
        int ps
)
{
    int pt_index = threadIdx.y + blockIdx.y*blockDim.y;
    if(pt_index>=pn)
        return;

    int* idxs=&nidxs[nidxs_bgs[pt_index]];
    float* covar=&covars[pt_index*9];
    int nn_size=nidxs_lens[pt_index];

    float sx=0.f,sy=0.f,sz=0.f;
    float sxy=0.f,sxz=0.f,syz=0.f;
    float sxx=0.f,syy=0.f,szz=0.f;
    for(int i=0;i<nn_size;i++)
    {
        int cur_pt_index=idxs[i];
        float x=pts[cur_pt_index*ps];
        float y=pts[cur_pt_index*ps+1];
        float z=pts[cur_pt_index*ps+2];
        sx+=x;sy+=y;sz+=z;
        sxy+=x*y;sxz+=x*z;syz+=y*z;
        sxx+=x*x;syy+=y*y;szz+=z*z;
    }

    int n=nn_size;
    float cxx=sxx/n-sx/n*sx/n;
    float cyy=syy/n-sy/n*sy/n;
    float czz=szz/n-sz/n*sz/n;
    float cxy=sxy/n-sx/n*sy/n;
    float cxz=sxz/n-sx/n*sz/n;
    float cyz=syz/n-sy/n*sz/n;

    //normalize
    double norm=sqrt(cxx*cxx+cyy*cyy+czz*czz+cxy*cxy*2+cxz*cxz*2+cyz*cyz*2);
    if(norm>1e-6)
    {
        cxx/=norm;
        cyy/=norm;
        czz/=norm;
        cxy/=norm;
        cxz/=norm;
        cyz/=norm;
    }

    covar[0]=cxx;covar[1]=cxy;covar[2]=cxz;
    covar[3]=cxy;covar[4]=cyy;covar[5]=cyz;
    covar[6]=cxz;covar[7]=cyz;covar[8]=czz;
}



void computeCovarsGPUImpl(
        float* pts,             // [pn,ps]
        int* nidxs,             // [csum]
        int* nidxs_lens,        // [pn]
        int* nidxs_bgs,         // [pn]
        float* covars,          // [pn,9]
        int pn,
        int ps,
        int csum,
        int gpu_index
)
{
    gpuErrchk(cudaSetDevice(gpu_index))
    int block_num=pn/1024;
    if(pn%1024>0) block_num++;
    dim3 block_dim(1,block_num);
    dim3 thread_dim(1,1024);

    float* d_pts;
    gpuErrchk(cudaMalloc((void**)&d_pts, pn * ps* sizeof(float)))
    gpuErrchk(cudaMemcpy(d_pts, pts, pn * ps* sizeof(float),cudaMemcpyHostToDevice))

    int* d_nidxs,*d_nidxs_lens,*d_nidxs_bgs;
    gpuErrchk(cudaMalloc((void**)&d_nidxs, csum * sizeof(int)))
    gpuErrchk(cudaMemcpy(d_nidxs, nidxs, csum * sizeof(int),cudaMemcpyHostToDevice))
    gpuErrchk(cudaMalloc((void**)&d_nidxs_lens, pn * sizeof(int)))
    gpuErrchk(cudaMemcpy(d_nidxs_lens, nidxs_lens, pn * sizeof(int),cudaMemcpyHostToDevice))
    gpuErrchk(cudaMalloc((void**)&d_nidxs_bgs, pn * sizeof(int)))
    gpuErrchk(cudaMemcpy(d_nidxs_bgs, nidxs_bgs, pn * sizeof(int),cudaMemcpyHostToDevice))

    float* d_covars;
    gpuErrchk(cudaMalloc((void**)&d_covars, pn * 9 * sizeof(float)))

    computeCovarsGPUKernel <<<block_dim,thread_dim>>>(d_pts,d_nidxs,d_nidxs_lens,d_nidxs_bgs,d_covars,pn,ps);

    gpuErrchk(cudaMemcpy(covars, d_covars, pn * 9 * sizeof(float),cudaMemcpyDeviceToHost));

    cudaFree(d_pts);
    cudaFree(d_nidxs);
    cudaFree(d_nidxs_lens);
    cudaFree(d_nidxs_bgs);
    cudaFree(d_covars);
}
