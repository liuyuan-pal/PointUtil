//
// Created by pal on 18-1-14.
//

#ifndef _CUDA_COMMON_H
#define _CUDA_COMMON_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cstdio>

#define HANDLE_ERROR(func,message) if((func)!=cudaSuccess) { printf("%s \n",message);   return; }

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#endif //_CUDA_COMMON_H
