#include "CudaUtil.h"


__global__
void gridDownSampleKernel(
        float* pts,
        unsigned short* grid_idxs,
        int pt_num,
        int pt_stride,
        float sample_stride,
        float min_x,
        float min_y,
        float min_z
)
{
    int pt_index = threadIdx.y + blockIdx.y*blockDim.y;
    if(pt_index>=pt_num)
        return;

    float x=pts[pt_index*pt_stride];
    float y=pts[pt_index*pt_stride+1];
    float z=pts[pt_index*pt_stride+2];

    grid_idxs[pt_index*3] = floor((x-min_x)/sample_stride);
    grid_idxs[pt_index*3+1] = floor((y-min_y)/sample_stride);
    grid_idxs[pt_index*3+2] = floor((z-min_z)/sample_stride);

}

void gridDownSampleIdxMap(
        float* h_pts,
        unsigned short* h_grid_idxs,
        int pt_num,
        int pt_stride,
        float sample_stride,
        float min_x,
        float min_y,
        float min_z
)
{

    int block_num=pt_num/1024;
    if(pt_num%1024>0) block_num++;
    dim3 block_dim(1,block_num);
    dim3 thread_dim(1,1024);

    float* d_pts;
    unsigned short* d_grid_idxs;
    gpuErrchk(cudaMalloc((void**)&d_pts, pt_num * pt_stride * sizeof(float)))
    gpuErrchk(cudaMalloc((void**)&d_grid_idxs, pt_num * 3 * sizeof(unsigned short)))
    gpuErrchk(cudaMemcpy(d_pts, h_pts, pt_num * pt_stride * sizeof(float), cudaMemcpyHostToDevice))
    gridDownSampleKernel<<<block_dim,thread_dim>>>(
            d_pts,d_grid_idxs,pt_num,pt_stride,sample_stride,min_x,min_y,min_z
    );
    gpuErrchk(cudaMemcpy(h_grid_idxs, d_grid_idxs, pt_num * 3 * sizeof(unsigned short), cudaMemcpyDeviceToHost))

    cudaFree(d_pts);
    cudaFree(d_grid_idxs);
}