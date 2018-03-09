#include "CudaUtil.h"
#include "../../../../usr/local/cuda/include/host_defines.h"
#include <cmath>

__global__
void interpolateKernel(
    float* sxyzs,   // [spn,3]
    float* sprobs,  // [spn,cn]
    float* qxyzs,   // [qpn,3]
    float* qprobs,  // [qpn,cn]
    int* nidxs,
    int* nidxs_lens,// [qpn]
    int* nidxs_bgs, // [qpn]
    int spn,
    int qpn,
    int cn,
    float ratio
)
{
    int qpi = threadIdx.x + blockIdx.x*blockDim.x;
    if(qpi>=qpn) return;

    // compute distance
    float sum_exp_neg=0.0;
    float qx=qxyzs[qpi*3+0];
    float qy=qxyzs[qpi*3+1];
    float qz=qxyzs[qpi*3+2];
    int nbg=nidxs_bgs[qpi];
    int nn=nidxs_lens[qpi];
    int ned=nn+nbg;
    for(int ni=nbg;ni<ned;ni++)
    {
        float x=sxyzs[nidxs[ni]*3+0];
        float y=sxyzs[nidxs[ni]*3+1];
        float z=sxyzs[nidxs[ni]*3+2];
        float dist=sqrt((qx-x)*(qx-x)+(qy-y)*(qy-y)+(qz-z)*(qz-z));
        sum_exp_neg+=exp(-dist*ratio);
    }

    float* qprobs_p=&qprobs[qpi*cn];
    for(int ci=0;ci<cn;ci++)
        qprobs_p[ci]=0.f;

    for(int ni=nbg;ni<ned;ni++)
    {
        float x=sxyzs[nidxs[ni]*3+0];
        float y=sxyzs[nidxs[ni]*3+1];
        float z=sxyzs[nidxs[ni]*3+2];
        float dist=sqrt((qx-x)*(qx-x)+(qy-y)*(qy-y)+(qz-z)*(qz-z));
        float w=exp(-dist*ratio)/sum_exp_neg;
        for(int ci=0;ci<cn;ci++)
            qprobs_p[ci]+=w*sprobs[nidxs[ni]*cn+ci];
    }
}

void interpolateImpl(
        float* h_sxyzs,   // [spn,3]
        float* h_sprobs,  // [spn,cn]
        float* h_qxyzs,   // [qpn,3]
        float* h_qprobs,  // [qpn,cn]
        int* h_nidxs,
        int* h_nidxs_lens,// [qpn]
        int* h_nidxs_bgs, // [qpn]
        int spn,
        int qpn,
        int cn,
        int nn,
        float ratio,
        int gpu_id
)
{
    gpuErrchk(cudaSetDevice(gpu_id))
    int block_num=qpn/1024;
    if(qpn%1024>0) block_num++;
    dim3 block_dim(block_num);
    dim3 thread_dim(1024);

    float* d_syxzs,*d_sprobs,*d_qxyzs,*d_qprobs;
    gpuErrchk(cudaMalloc((void**)&d_syxzs, spn * 3 * sizeof(float)))
    gpuErrchk(cudaMalloc((void**)&d_sprobs, spn * cn * sizeof(float)))
    gpuErrchk(cudaMalloc((void**)&d_qxyzs, qpn * 3 * sizeof(float)))
    gpuErrchk(cudaMalloc((void**)&d_qprobs, qpn * cn * sizeof(float)))
    gpuErrchk(cudaMemcpy(d_syxzs, h_sxyzs, spn * 3 * sizeof(float), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_sprobs, h_sprobs, spn * cn * sizeof(float), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_qxyzs, h_qxyzs, qpn * 3 * sizeof(float), cudaMemcpyHostToDevice))

    int* d_nidxs,*d_nidxs_lens,*d_nidxs_bgs;
    gpuErrchk(cudaMalloc((void**)&d_nidxs, nn * sizeof(int)))
    gpuErrchk(cudaMalloc((void**)&d_nidxs_lens, qpn * sizeof(int)))
    gpuErrchk(cudaMalloc((void**)&d_nidxs_bgs, qpn * sizeof(int)))
    gpuErrchk(cudaMemcpy(d_nidxs, h_nidxs, nn * sizeof(int), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_nidxs_lens, h_nidxs_lens, qpn * sizeof(int), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_nidxs_bgs, h_nidxs_bgs, qpn * sizeof(int), cudaMemcpyHostToDevice))

    interpolateKernel<<<block_dim,thread_dim>>>
            (d_syxzs,d_sprobs,d_qxyzs,d_qprobs,d_nidxs,d_nidxs_lens,d_nidxs_bgs,spn,qpn,cn,ratio);


    gpuErrchk(cudaMemcpy(h_qprobs, d_qprobs, qpn * cn * sizeof(float), cudaMemcpyDeviceToHost))

    cudaFree(d_syxzs);
    cudaFree(d_sprobs);
    cudaFree(d_qxyzs);
    cudaFree(d_qprobs);
    cudaFree(d_nidxs);
    cudaFree(d_nidxs_lens);
    cudaFree(d_nidxs_bgs);
}