#include <vector>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include "PointCloudUtil.h"

void sampleRotatedBlockGatherPoints(
        float *points,         // [point_num,pt_stride] xyz rgb ...
        float *retain_origin,  // [retain_num,2] xy
        bool *result,          // [point_num,retain_num] 0 or 1
        int point_stride,
        int point_num,
        int retain_num,
        float block_axis_xx,
        float block_axis_xy,
        float block_axis_yx,
        float block_axis_yy,
        float block_size,
        float min_x,
        float min_y,
        int gpu_index
);

std::vector<std::vector<int> >
sampleRotatedBlockGPUImpl(
        float *points,
        int pt_num,
        int pt_stride,
        float sample_stride,
        float block_size,
        float rot_angle,
        int gpu_index
)
{
    // float rot_angle=float(rand()%90)/180*M_PI;
    // rot_angle_out=-rot_angle;
    float block_axis_xx=cos(rot_angle);
    float block_axis_xy=sin(rot_angle);
    float block_axis_yx=-block_axis_xy;
    float block_axis_yy=block_axis_xx;

    float min_x,min_y,min_z,max_x,max_y,max_z;
    findMinimum(points,pt_num,pt_stride,min_x,min_y,min_z);
    findMaximum(points,pt_num,pt_stride,max_x,max_y,max_z);
    float len_x=max_x-min_x,len_y=max_y-min_y;

    //std::cout<<"begin\n";
    std::vector<float> sample_origin;
    float x=-block_size;
    while(x<=len_x)
    {
        x+=sample_stride;
        float y=-block_size;
        while(y<=len_y)
        {
            y+=sample_stride;
            sample_origin.push_back(x);
            sample_origin.push_back(y);
        }
    }

    //std::cout<<"validate\n";
    int origin_num=sample_origin.size()/2;

    bool* result=new bool[pt_num*origin_num];
    sampleRotatedBlockGatherPoints(points, sample_origin.data(), result,
                                   pt_stride, pt_num, origin_num,
                                   block_axis_xx, block_axis_xy,
                                   block_axis_yx, block_axis_yy, block_size,
                                   min_x, min_y,
                                   gpu_index);


    std::vector<std::vector<int> > block_idxs(origin_num);

    //begin=clock();
    for(int i=0;i<pt_num;i++)
    {
        for(int j=0;j<origin_num;j++)
        {
            if(result[i*origin_num+j])
                block_idxs[j].push_back(i);
        }
    }

    //std::cout<<"gather cost "<<float(clock()-begin)/CLOCKS_PER_SEC<<"s \n";
    //std::cout<<"gather\n";

    delete [] result;
    return block_idxs;
}


std::vector<std::vector<int> >
sampleRotatedBlockGatherPointsGPUImpl(
        float *points,
        int pt_num,
        int pt_stride,
        float block_size,
        float *bg_list,
        int bg_num,
        int gpu_index
)
{
//    std::cout<<"here\n";
    float min_x,min_y,min_z;
    findMinimum(points,pt_num,pt_stride,min_x,min_y,min_z);

//    std::cout<<"here\n";
//    std::cout<<pt_num<<" "<<bg_num<<std::endl;
    bool* result=new bool[pt_num*bg_num];
//    std::cout<<"here\n";
    sampleRotatedBlockGatherPoints(points, bg_list, result,
                                   pt_stride, pt_num, bg_num,
                                   1.f, 0.f, 0.f, 1.f, block_size,
                                   min_x, min_y, gpu_index);


//    std::cout<<"here\n";
    std::vector<std::vector<int> > block_idxs(bg_num);

    //begin=clock();
    for(int i=0;i<pt_num;i++)
    {
        for(int j=0;j<bg_num;j++)
        {
            if(result[i*bg_num+j])
                block_idxs[j].push_back(i);
        }
    }

//    std::cout<<"here\n";
    std::vector<std::vector<int> > refined_block_idxs;
    for(int j=0;j<bg_num;j++)
    {
        if(block_idxs[j].size()>0)
            refined_block_idxs.push_back(block_idxs[j]);
    }

//    std::cout<<"here\n";
    //std::cout<<"gather cost "<<float(clock()-begin)/CLOCKS_PER_SEC<<"s \n";
    //std::cout<<"gather\n";

    delete [] result;
    return refined_block_idxs;
}