//
// Created by pal on 18-1-25.
//

#include <vector>
#include <tuple>
#include <unordered_map>
#include <string>
#include <iostream>
#include <sstream>
#include <limits>
#include "PointCloudUtil.h"

void gridDownSampleIdxMap(
        float* h_pts,
        unsigned short* h_grid_idxs,
        int pt_num,
        int pt_stride,
        float sample_stride,
        float min_x,
        float min_y,
        float min_z
);

void gridDownSample(
        float* pts,
        int pt_num,
        int pt_stride,
        float sample_stride,
        std::vector<int>& downsample_indices
)
{
    //time_t begin=clock();
    unsigned short* grid_idxs=new unsigned short[pt_num*3];
    float min_x,min_y,min_z;
    findMinimum(pts,pt_num,pt_stride,min_x,min_y,min_z);
    gridDownSampleIdxMap(pts,grid_idxs,pt_num,pt_stride,sample_stride,min_x,min_y,min_z);
    //std::cout<<"gpu map "<<float(clock()-begin)/CLOCKS_PER_SEC<<std::endl;

    //begin=clock();
    std::unordered_map<unsigned long long, std::vector<int>> map;
    for(int i=0;i<pt_num;i++)
    {
        unsigned long long x=grid_idxs[i*3];
        unsigned long long y=grid_idxs[i*3+1];
        unsigned long long z=grid_idxs[i*3+2];
        unsigned long long idx=0;
        idx=(idx|x)|(y<<16|z<<32);

        auto it=map.find(idx);
        if(it!=map.end())
        {
            it->second.push_back(i);
        }
        else
        {
            std::vector<int> pt_idxs;
            pt_idxs.push_back(i);
            map[idx]=pt_idxs;
        }
    }
    //std::cout<<"sort map "<<float(clock()-begin)/CLOCKS_PER_SEC<<std::endl;

    //begin=clock();
    for(auto it=map.begin();it!=map.end();it++)
    {
        int select_idx=rand()%it->second.size();
        downsample_indices.push_back(it->second[select_idx]);
    }
    //std::cout<<"push back "<<float(clock()-begin)/CLOCKS_PER_SEC<<std::endl;

    delete[] grid_idxs;
}


void gridDownSampleV2(
        float* pts,
        int pt_num,
        int pt_stride,
        float sample_stride,
        std::vector<int>& ds_idxs,
        std::vector<int>& ds_gidxs
)
{
    time_t begin=clock();
    unsigned short* grid_idxs=new unsigned short[pt_num*3];
    float min_x,min_y,min_z;
    findMinimum(pts,pt_num,pt_stride,min_x,min_y,min_z);
    gridDownSampleIdxMap(pts,grid_idxs,pt_num,pt_stride,sample_stride,min_x,min_y,min_z);
    //std::cout<<"gpu map "<<float(clock()-begin)/CLOCKS_PER_SEC<<std::endl;

    //begin=clock();
    std::unordered_map<unsigned long long, int> map;
    int ds_pt_num=0;
    ds_gidxs.resize(pt_num);
    std::vector<int> gsize;
    for(int i=0;i<pt_num;i++)
    {
        unsigned long long x=grid_idxs[i*3];
        unsigned long long y=grid_idxs[i*3+1];
        unsigned long long z=grid_idxs[i*3+2];
        unsigned long long idx=0;
        idx=(idx|x)|(y<<16|z<<32);

        auto it=map.find(idx);
        if(it!=map.end())
        {
            ds_gidxs[i]=it->second;
            gsize[it->second]+=1;
        }
        else
        {
            map[idx]=ds_pt_num;
            ds_gidxs[i]=ds_pt_num;
            gsize.push_back(1);
            ds_pt_num++;
        }
    }
    //std::cout<<"sort map "<<float(clock()-begin)/CLOCKS_PER_SEC<<std::endl;

    //begin=clock();
    srand(time(0));
    std::vector<int> retain_idxs(ds_pt_num);
    for(int i=0;i<ds_pt_num;i++)
        retain_idxs[i]=(rand()%(gsize[i]));
    std::vector<int> gloc(ds_pt_num);
    ds_idxs.resize(ds_pt_num);
    for(int i=0;i<pt_num;i++)
    {
        int gid=ds_gidxs[i];
        if(gloc[gid]==retain_idxs[gid])
            ds_idxs[gid]=i;
        gloc[gid]+=1;
    }
    //std::cout<<"push cost "<<float(clock()-begin)/CLOCKS_PER_SEC<<std::endl;
    delete[] grid_idxs;
}