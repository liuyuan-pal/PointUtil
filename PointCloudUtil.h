//
// Created by pal on 18-2-27.
//

#include <limits>
#include <iostream>

inline void findMinimum(
        float* pts,
        int pt_num,
        int pt_stride,
        float&min_x,float&min_y,float& min_z
)
{
    min_x=std::numeric_limits<float>::max(),
    min_y=std::numeric_limits<float>::max(),
    min_z=std::numeric_limits<float>::max();
    for(int i=0;i<pt_num;i++)
    {
        min_x=std::min(min_x,pts[i*pt_stride]);
        min_y=std::min(min_y,pts[i*pt_stride+1]);
        min_z=std::min(min_z,pts[i*pt_stride+2]);
    }
}

inline void findMaximum(
        float* pts,
        int pt_num,
        int pt_stride,
        float&max_x,float&max_y,float& max_z
)
{
    max_x=-std::numeric_limits<float>::max(),
    max_y=-std::numeric_limits<float>::max(),
    max_z=-std::numeric_limits<float>::max();
    for(int i=0;i<pt_num;i++)
    {
        max_x=std::max(max_x,pts[i*pt_stride]);
        max_y=std::max(max_y,pts[i*pt_stride+1]);
        max_z=std::max(max_z,pts[i*pt_stride+2]);
    }
}