#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#define FLANN_USE_CUDA
#include <flann/flann.hpp>
#include "ConvertUtil.h"
#include <iostream>
using namespace std;

namespace bp=boost::python;
namespace bpn=boost::python::numpy;

inline void
fillQueryPoints(
        const float* pts_data,
        const int* idx_data,
        std::vector<float>& qpts,
        int idx_num
)
{
    qpts.resize(idx_num*3);
    for(int i=0;i<idx_num;i++)
    {
        int pt_idx=idx_data[i];
        memcpy(&qpts[i*3],&pts_data[pt_idx*3],3*sizeof(float));
    }
}

PyObject* findNeighborRadiusAllGPU(const bpn::ndarray& pts,float radius,int leaf_size=15)
{
    assert(pts.shape(1)==3);
    assert(pts.get_flags()&bpn::ndarray::C_CONTIGUOUS); //must be contiguous
    assert(std::strcmp(bp::extract<const char *>(bp::str(pts.get_dtype())),"float32")==0);

    const long int pt_num=pts.shape(0);
    float * pts_data = reinterpret_cast<float*>(pts.get_data());
    PyObject* list;
    std::vector<std::vector<int> > indices;
    std::vector<std::vector<float> > dists;

    Py_BEGIN_ALLOW_THREADS
    flann::Matrix<float> dataset(pts_data,pt_num,3);
    flann::KDTreeCuda3dIndex<flann::L2_Simple<float> > index(dataset,flann::KDTreeCuda3dIndexParams(leaf_size));
    index.buildIndex();

    flann::SearchParams search_params;
    search_params.sorted=false;
    search_params.eps=0.f;
    index.radiusSearch(dataset,indices,dists,radius*radius,search_params);  // due to L2_simple dont sqrt the distance
    Py_END_ALLOW_THREADS
    list= vec2DToListNDArray2DInt(indices);

    return list;
}

PyObject* findNeighborRadiusPartGPU(const bpn::ndarray& pts,const bpn::ndarray& idx,float radius,int leaf_size=15)
{
    assert(pts.shape(1)==3);
    assert(pts.get_flags()&bpn::ndarray::C_CONTIGUOUS); //must be contiguous
    assert(std::strcmp(bp::extract<const char *>(bp::str(pts.get_dtype())),"float32")==0);

    assert(idx.get_flags()&bpn::ndarray::C_CONTIGUOUS); //must be contiguous
    assert(std::strcmp(bp::extract<const char *>(bp::str(idx.get_dtype())),"int32")==0);

    const long int pt_num=pts.shape(0);
    const long int idx_num=idx.shape(0);
    auto pts_data = reinterpret_cast<float*>(pts.get_data());
    auto idx_data = reinterpret_cast<int*>(idx.get_data());
    PyObject* list;

    std::vector<std::vector<int> > indices;
    std::vector<std::vector<float> > dists;
    Py_BEGIN_ALLOW_THREADS
    flann::Matrix<float> dataset(pts_data,pt_num,3);
    flann::KDTreeCuda3dIndex<flann::L2_Simple<float> > index(dataset,flann::KDTreeCuda3dIndexParams(leaf_size));
    index.buildIndex();

    std::vector<float> qpts;
    fillQueryPoints(pts_data,idx_data,qpts,idx_num);
    flann::Matrix<float> query_dataset(qpts.data(),idx_num,3);

    flann::SearchParams search_params;
    search_params.sorted=false;
    search_params.eps=0.f;
    index.radiusSearch(query_dataset,indices,dists,radius*radius,search_params);
    Py_END_ALLOW_THREADS
    list= vec2DToListNDArray2DInt(indices);

    return list;
}

PyObject* findNeighborRadiusAllCPU(const bpn::ndarray& pts,float radius,int leaf_size=15)
{
    assert(pts.shape(1)==3);
    assert(pts.get_flags()&bpn::ndarray::C_CONTIGUOUS); //must be contiguous
    assert(std::strcmp(bp::extract<const char *>(bp::str(pts.get_dtype())),"float32")==0);

    const long int pt_num=pts.shape(0);
    float * pts_data = reinterpret_cast<float*>(pts.get_data());
    PyObject* list;

    std::vector<std::vector<int> > indices;
    std::vector<std::vector<float> > dists;
    Py_BEGIN_ALLOW_THREADS
    flann::Matrix<float> dataset(pts_data,pt_num,3);
    flann::Index<flann::L2_Simple<float> > index(dataset,flann::KDTreeSingleIndexParams(leaf_size));
    index.buildIndex();

    flann::SearchParams search_params;
    search_params.cores=4;
    search_params.sorted=false;
    search_params.eps=0.f;
    index.radiusSearch(dataset,indices,dists,radius*radius,search_params);
    Py_END_ALLOW_THREADS
    list= vec2DToListNDArray2DInt(indices);

    return list;
}

PyObject* findNeighborRadiusPartCPU(const bpn::ndarray& pts,const bpn::ndarray& idx,float radius,int leaf_size=15)
{
    assert(pts.shape(1)==3);
    assert(pts.get_flags()&bpn::ndarray::C_CONTIGUOUS); //must be contiguous
    assert(std::strcmp(bp::extract<const char *>(bp::str(pts.get_dtype())),"float32")==0);

    assert(idx.get_flags()&bpn::ndarray::C_CONTIGUOUS); //must be contiguous
    assert(std::strcmp(bp::extract<const char *>(bp::str(idx.get_dtype())),"int32")==0);

    PyObject* list;
    const long int pt_num=pts.shape(0);
    const long int idx_num=idx.shape(0);
    auto pts_data = reinterpret_cast<float*>(pts.get_data());
    auto idx_data = reinterpret_cast<int*>(idx.get_data());

    std::vector<std::vector<int> > indices;
    std::vector<std::vector<float> > dists;
    Py_BEGIN_ALLOW_THREADS
    flann::Matrix<float> dataset(pts_data,pt_num,3);
    flann::KDTreeSingleIndex<flann::L2_Simple<float> > index(dataset,flann::KDTreeSingleIndexParams(leaf_size));
    index.buildIndex();

    std::vector<float> qpts;
    fillQueryPoints(pts_data,idx_data,qpts,idx_num);
    flann::Matrix<float> query_dataset(qpts.data(),idx_num,3);

    flann::SearchParams search_params;
    search_params.cores=4;
    search_params.sorted=false;
    search_params.eps=0.f;
    index.radiusSearch(query_dataset,indices,dists,radius*radius,search_params);
    Py_END_ALLOW_THREADS
    list= vec2DToListNDArray2DInt(indices);

    return list;
}

void gridDownSample(
        float* pts,
        int pt_num,
        int pt_stride,
        float sample_stride,
        std::vector<int>& ds_idxs,
        int gpu_id
);

void gridDownSampleV2(
        float* pts,
        int pt_num,
        int pt_stride,
        float sample_stride,
        std::vector<int>& ds_idxs,
        std::vector<int>& ds_gidxs,
        int gpu_id
);

void computeGridIdx(
        float* pts,
        int pt_num,
        int pt_stride,
        float sample_stride,
        std::vector<int>& ds_gidxs,
        int gpu_id
);

static PyObject*
gridDownsampleGPU(const bpn::ndarray &pts, float sample_stride, bool output_gidxs = false,int gpu_id=0)
{
    assert(pts.shape(1)>=3);
    assert(pts.get_flags()&bpn::ndarray::C_CONTIGUOUS); //must be contiguous
    assert(std::strcmp(bp::extract<const char *>(bp::str(pts.get_dtype())),"float32")==0);

    float* pts_data=reinterpret_cast<float*>(pts.get_data());
    int pt_num=pts.shape(0),pt_stride=pts.shape(1);
    std::vector<int> ds_idxs;
    PyObject* result;
    if(!output_gidxs)
    {
        Py_BEGIN_ALLOW_THREADS
        gridDownSample(pts_data,pt_num,pt_stride,sample_stride,ds_idxs,gpu_id);
        Py_END_ALLOW_THREADS
        result=vec1DToNDArray1DInt(ds_idxs);
    }
    else
    {
        std::vector<int> ds_gidxs;
        Py_BEGIN_ALLOW_THREADS
        gridDownSampleV2(pts_data,pt_num,pt_stride,sample_stride,ds_idxs,ds_gidxs,gpu_id);
        Py_END_ALLOW_THREADS
        PyObject* ds_idxs_arr=vec1DToNDArray1DInt(ds_idxs);
        PyObject* ds_gidxs_arr=vec1DToNDArray1DInt(ds_gidxs);
        bp::list* list=new bp::list();
        list->append(bp::handle<>(ds_idxs_arr));
        list->append(bp::handle<>(ds_gidxs_arr));
        result=list->ptr();
    }
    return result;
}

std::vector<std::vector<int> >
sampleRotatedBlockGPUImpl(
        float *points,
        int pt_num,
        int pt_stride,
        float sample_stride,
        float block_size,
        float rot_angle,
        int gpu_index
);

static PyObject*
sampleRotatedBlockGPU(const bpn::ndarray &pts, float sample_stride, float block_size,float rotate_angle, int gpu_index=0)
{
    assert(pts.shape(1)>=3);
    assert(pts.get_flags()&bpn::ndarray::C_CONTIGUOUS); //must be contiguous
    assert(std::strcmp(bp::extract<const char *>(bp::str(pts.get_dtype())),"float32")==0);

    float* pts_data=reinterpret_cast<float*>(pts.get_data());
    int pt_num=pts.shape(0),pt_stride=pts.shape(1);
    PyObject* list;

    std::vector<std::vector<int> > block_idxs;
    Py_BEGIN_ALLOW_THREADS
    block_idxs=sampleRotatedBlockGPUImpl(pts_data, pt_num, pt_stride, sample_stride, block_size, rotate_angle, gpu_index);
    Py_END_ALLOW_THREADS
    list=vec2DToListNDArray2DInt(block_idxs);

    return list;

}

void computeCovarsGPUImpl(
        float* pts,             // [pn,ps]
        int* nidxs,             // [csum]
        int* nidxs_lens,        // [fn]
        int* nidxs_bgs,         // [fn]
        float* covars,          // [fn,9]
        int pn,
        int fn,
        int ps,
        int csum,
        int gpu_index
);

static PyObject*
computeCovarsGPU(
        const bpn::ndarray& ixyz,            // [pn,ps]
        const bpn::ndarray& inidxs,          // [csum]
        const bpn::ndarray& inidxs_lens,     // [fn]
        const bpn::ndarray& inidxs_bgs,      // [fn]
        int gpu_index=0
)
{
    assert(ixyz.get_nd()==2);
    assert(ixyz.get_flags()&bpn::ndarray::C_CONTIGUOUS);
    assert(std::strcmp(bp::extract<const char *>(bp::str(ixyz.get_dtype())),"float32")==0);
    int pn=ixyz.shape(0);
    int ps=ixyz.shape(1);

    assert(inidxs.get_nd()==1);
    assert(inidxs.get_flags()&bpn::ndarray::C_CONTIGUOUS);
    assert(std::strcmp(bp::extract<const char *>(bp::str(inidxs.get_dtype())),"int32")==0);
    int csum=inidxs.shape(0);

    assert(inidxs_lens.get_nd()==1);
    assert(inidxs_lens.get_flags()&bpn::ndarray::C_CONTIGUOUS);
    assert(std::strcmp(bp::extract<const char *>(bp::str(inidxs_lens.get_dtype())),"int32")==0);
    int fn=inidxs_lens.shape(0);

    assert(inidxs_bgs.get_nd()==1);
    assert(inidxs_bgs.shape(0)==fn);
    assert(inidxs_bgs.get_flags()&bpn::ndarray::C_CONTIGUOUS);
    assert(std::strcmp(bp::extract<const char *>(bp::str(inidxs_bgs.get_dtype())),"int32")==0);

    float* xyz_data = reinterpret_cast<float*>(ixyz.get_data());
    int* inidxs_data = reinterpret_cast<int*>(inidxs.get_data());
    int* inidxs_lens_data = reinterpret_cast<int*>(inidxs_lens.get_data());
    int* inidxs_bgs_data = reinterpret_cast<int*>(inidxs_bgs.get_data());

    PyObject* covars=npArray2Df(pn,9);

    Py_BEGIN_ALLOW_THREADS
    float* covars_data= reinterpret_cast<float*>(PyArray_DATA((PyArrayObject*)covars));
    computeCovarsGPUImpl(xyz_data,inidxs_data,inidxs_lens_data,inidxs_bgs_data,covars_data,pn,fn,ps,csum,gpu_index);
    Py_END_ALLOW_THREADS

    return covars;
}

void sortVoxelGPUImpl(
        float *xyz,                 // [pn,ps]
        int pn,
        int ps,
        float vs,                   // voxel lens
        std::vector<int> &sidxs,    // [pn]
        std::vector<int> &vlens,    // [vn]
        int gpu_id
);

static PyObject*
sortVoxelGPU(const bpn::ndarray &pts, float vs, int gpu_id=0)
{
    assert(pts.shape(1)>=3);
    assert(pts.get_flags()&bpn::ndarray::C_CONTIGUOUS); //must be contiguous
    assert(std::strcmp(bp::extract<const char *>(bp::str(pts.get_dtype())),"float32")==0);

    float* pts_data=reinterpret_cast<float*>(pts.get_data());
    int pn=pts.shape(0),ps=pts.shape(1);
    std::vector<int> sidxs,vlens;
    Py_BEGIN_ALLOW_THREADS
    sortVoxelGPUImpl(pts_data,pn,ps,vs,sidxs,vlens,gpu_id);
    Py_END_ALLOW_THREADS

    PyObject* sidxs_arr=vec1DToNDArray1DInt(sidxs);
    PyObject* vlens_arr=vec1DToNDArray1DInt(vlens);
    bp::list* list=new bp::list();
    list->append(bp::handle<>(sidxs_arr));
    list->append(bp::handle<>(vlens_arr));

    return list->ptr();
}


void computeCenterDiffCPUImpl(
        float *xyz,                 // [pn,ps]
        int* vlens,                 // [vn]
        int pn,
        int ps,
        int vn,
        std::vector<float>& dxyz,   // [pn,3]
        std::vector<float>& cxyz    // [vn,3]
);

static PyObject*
computeCenterDiffCPU(
        const bpn::ndarray &pts,
        const bpn::ndarray &vlens
)
{
    assert(pts.shape(1) >= 3);
    assert(pts.get_flags() & bpn::ndarray::C_CONTIGUOUS); //must be contiguous
    assert(std::strcmp(bp::extract<const char *>(bp::str(pts.get_dtype())), "float32") == 0);

    assert(vlens.get_nd()==1);
    assert(vlens.get_flags()&bpn::ndarray::C_CONTIGUOUS);
    assert(std::strcmp(bp::extract<const char *>(bp::str(vlens.get_dtype())),"int32")==0);

    float* pts_data=reinterpret_cast<float*>(pts.get_data());
    int* vlens_data=reinterpret_cast<int*>(vlens.get_data());
    int pn=pts.shape(0),ps=pts.shape(1);
    int vn=vlens.shape(0);

    std::vector<float> dxyz,cxyz;
    computeCenterDiffCPUImpl(pts_data,vlens_data,pn,ps,vn,dxyz,cxyz);

    PyObject* dxyz_arr=vec1DToNDArray2D(dxyz,pn,3);
    PyObject* cxyz_arr=vec1DToNDArray2D(cxyz,vn,3);
    bp::list* list=new bp::list();
    list->append(bp::handle<>(dxyz_arr));
    list->append(bp::handle<>(cxyz_arr));

    return list->ptr();
}

void adjustPointsMemoryCPUImpl(
        // k=vlens_c2n is the i th point of lvl next layer corresponds to k points of lvl cur layer
        int* vlens_c2n,     // [vn] lvl cur -> lvl next
        // k=sidxs_nad[i] is the i th point in lvl next layer is map to the k th location
        int* sidxs_nad,     // [vn] lvl next location adjust
        int vn,
        int pn,
        std::vector<int>& sidxs_cad,  // [pn] lvl cur this adjust
        std::vector<int>& vlens_c2n_adj
);

static PyObject*
adjustPointsMemoryCPU(
        const bpn::ndarray &vlens_c2n,
        const bpn::ndarray &sidxs_nad,
        int pn
)
{
    assert(vlens_c2n.get_flags() & bpn::ndarray::C_CONTIGUOUS); //must be contiguous
    assert(std::strcmp(bp::extract<const char *>(bp::str(vlens_c2n.get_dtype())), "int32") == 0);
    int vn=vlens_c2n.shape(0);

    assert(sidxs_nad.shape(0) == vn);
    assert(sidxs_nad.get_flags()&bpn::ndarray::C_CONTIGUOUS);
    assert(std::strcmp(bp::extract<const char *>(bp::str(sidxs_nad.get_dtype())),"int32")==0);


    int* vlens_c2n_data=reinterpret_cast<int*>(vlens_c2n.get_data());
    int* sidxs_nad_data=reinterpret_cast<int*>(sidxs_nad.get_data());

    std::vector<int> sidxs_cad,vlens_c2n_adj;
    adjustPointsMemoryCPUImpl(vlens_c2n_data,sidxs_nad_data,vn,pn,sidxs_cad,vlens_c2n_adj);

    PyObject* sidxs_cad_arr=vec1DToNDArray1DInt(sidxs_cad);
    PyObject* vlens_c2n_adj_arr=vec1DToNDArray1DInt(vlens_c2n_adj);
    bp::list* list=new bp::list();
    list->append(bp::handle<>(sidxs_cad_arr));
    list->append(bp::handle<>(vlens_c2n_adj_arr));

    return list->ptr();
}


BOOST_PYTHON_FUNCTION_OVERLOADS(findNeighborRadiusAllGPUOverloads, findNeighborRadiusAllGPU, 2, 3);
BOOST_PYTHON_FUNCTION_OVERLOADS(findNeighborRadiusPartGPUOverloads, findNeighborRadiusPartGPU, 3, 4);
BOOST_PYTHON_FUNCTION_OVERLOADS(findNeighborRadiusAllCPUOverloads, findNeighborRadiusAllCPU, 2, 3);
BOOST_PYTHON_FUNCTION_OVERLOADS(findNeighborRadiusPartCPUOverloads, findNeighborRadiusPartCPU, 3, 4);

BOOST_PYTHON_FUNCTION_OVERLOADS(gridDownsampleGPUOverloads, gridDownsampleGPU, 2, 3);
BOOST_PYTHON_FUNCTION_OVERLOADS(sampleRotatedBlockGPUOverloads, sampleRotatedBlockGPU, 4, 5);
BOOST_PYTHON_FUNCTION_OVERLOADS(computeCovarsGPUOverloads, computeCovarsGPU, 4, 5);
BOOST_PYTHON_FUNCTION_OVERLOADS(sortVoxelGPUOverloads, sortVoxelGPU, 2, 3);

BOOST_PYTHON_MODULE(libPointUtil)
{
    _import_array();
    Py_Initialize();
    bpn::initialize();
    bp::def("findNeighborRadiusGPU",findNeighborRadiusAllGPU,findNeighborRadiusAllGPUOverloads());
    bp::def("findNeighborRadiusGPU",findNeighborRadiusPartGPU,findNeighborRadiusPartGPUOverloads());
    bp::def("findNeighborRadiusCPU",findNeighborRadiusAllCPU,findNeighborRadiusAllCPUOverloads());
    bp::def("findNeighborRadiusCPU",findNeighborRadiusPartCPU,findNeighborRadiusPartCPUOverloads());

    bp::def("gridDownsampleGPU", gridDownsampleGPU, gridDownsampleGPUOverloads());

    bp::def("sampleRotatedBlockGPU", sampleRotatedBlockGPU, sampleRotatedBlockGPUOverloads());

    bp::def("computeCovarsGPU", computeCovarsGPU, computeCovarsGPUOverloads());

    bp::def("sortVoxelGPU", sortVoxelGPU, sortVoxelGPUOverloads());

    bp::def("computeCenterDiffCPU", computeCenterDiffCPU);
    bp::def("adjustPointsMemoryCPU", adjustPointsMemoryCPU);
}
