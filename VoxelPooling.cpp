//
// Created by pal on 18-3-5.
//
#include <vector>
#include <unordered_map>
#include "PointCloudUtil.h"

void gridDownSampleIdxMap(
        float* h_pts,
        unsigned short* h_grid_idxs,
        int pt_num,
        int pt_stride,
        float sample_stride,
        float min_x,
        float min_y,
        float min_z,
        int gpu_id
);

void sortVoxelGPUImpl(
        float *xyz,                 // [pn,ps]
        int pn,
        int ps,
        float vs,                   // voxel lens
        std::vector<int> &sidxs,    // [pn]
        std::vector<int> &vlens,    // [vn]
        int gpu_id
)
{
    unsigned short* grid_idxs=new unsigned short[pn*3];
    float min_x,min_y,min_z;
    findMinimum(xyz,pn,ps,min_x,min_y,min_z);
    gridDownSampleIdxMap(xyz,grid_idxs,pn,ps,vs,min_x,min_y,min_z,gpu_id);

    std::unordered_map<unsigned long long, int> map;
    int vn=0;
    std::vector<int> ds_gidxs(pn);
    for(int i=0;i<pn;i++)
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
        }
        else
        {
            map[idx]=vn;
            ds_gidxs[i]=vn;
            vn++;
        }
    }

    std::vector<std::vector<int>> list_idxs(vn);
    for(int i=0;i<pn;i++)
        list_idxs[ds_gidxs[i]].push_back(i);

    // sidxs vlens
    sidxs.resize(pn);
    vlens.resize(vn);
    int* it=sidxs.begin().base();
    for(int i=0;i<vn;i++)
    {
        std::copy(list_idxs[i].begin(),list_idxs[i].end(),it);
        it+=list_idxs[i].size();
        vlens[i]=list_idxs[i].size();
    }

    delete [] grid_idxs;
}


void computeCenterDiffCPUImpl(
        float *xyz,                 // [pn,ps]
        int* vlens,                 // [vn]
        int pn,
        int ps,
        int vn,
        std::vector<float>& dxyz,   // [pn,3]
        std::vector<float>& cxyz    // [vn,3]
)
{
    cxyz.resize(vn*3);
    dxyz.resize(pn*3);
    float* vp=xyz,*dxyz_p=dxyz.begin().base();
    for(int i=0;i<vn;i++)
    {
        // compute center
        float* cp=vp;
        float cx=0,cy=0,cz=0;
        for(int j=0;j<vlens[i];j++)
        {
            cx+=cp[0];cy+=cp[1];cz+=cp[2];
            cp+=ps;
        }
        cx/=vlens[i];cy/=vlens[i];cz/=vlens[i];
        cxyz[i*3]=cx;cxyz[i*3+1]=cy;cxyz[i*3+2]=cz;

        // computer diff
        cp=vp;
        for(int j=0;j<vlens[i];j++)
        {
            dxyz_p[0]=cp[0]-cx;
            dxyz_p[1]=cp[1]-cy;
            dxyz_p[2]=cp[2]-cz;
            dxyz_p+=3;
            cp+=ps;
        }

        vp+=vlens[i]*ps;
    }
}

void adjustPointsMemoryCPUImpl(
        // k=vlens_c2n is the i th point of lvl next layer corresponds to k points of lvl cur layer
        int* vlens_c2n,     // [vn] lvl cur -> lvl next
        // k=sidxs_nad[i] is the i th point of lvl next next layer corresponds to the k th point of next layer
        int* sidxs_nad,     // [vn] lvl next location adjust
        int vn,
        int pn,
        std::vector<int>& sidxs_cad,  // [pn] lvl cur this adjust
        std::vector<int>& vlens_c2n_adj
)
{
    //compute inverse map
    std::vector<int> sidxs_nad_inv(vn);
    for(int i=0;i<vn;i++)
        sidxs_nad_inv[sidxs_nad[i]]=i;

    // compute new vlens
    vlens_c2n_adj.resize(vn);
    for(int i=0;i<vn;i++)
        vlens_c2n_adj[sidxs_nad_inv[i]]=vlens_c2n[i];

    // count begins
    std::vector<int> bg(vn+1);
    bg[0]=0;
    for(int i=0;i<vn;i++)
        bg[i+1]=bg[i]+vlens_c2n[i];

    // compute new sidxs
    sidxs_cad.resize(pn);
    int cur_bg=0;
    for(int i=0;i<vn;i++)
    {
        for(int j=0;j<vlens_c2n_adj[i];j++)
        {
            sidxs_cad[cur_bg+j]=bg[sidxs_nad[i]]+j;
        }
        cur_bg+=vlens_c2n_adj[i];
    }
}