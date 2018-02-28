#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#define FLANN_USE_CUDA
#include <flann/flann.hpp>
#include "ConvertUtil.h"
#include <iostream>

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

    Py_BEGIN_ALLOW_THREADS
    flann::Matrix<float> dataset(pts_data,pt_num,3);
    flann::KDTreeCuda3dIndex<flann::L2_Simple<float> > index(dataset,flann::KDTreeCuda3dIndexParams(leaf_size));
    index.buildIndex();

    std::vector<std::vector<int> > indices;
    std::vector<std::vector<float> > dists;
    flann::SearchParams search_params;
    search_params.sorted=false;
    index.radiusSearch(dataset,indices,dists,radius*radius,search_params);  // due to L2_simple dont sqrt the distance
    list= vec2DToListNDArray2DInt(indices);
    Py_END_ALLOW_THREADS

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

    Py_BEGIN_ALLOW_THREADS
    flann::Matrix<float> dataset(pts_data,pt_num,3);
    flann::KDTreeCuda3dIndex<flann::L2_Simple<float> > index(dataset,flann::KDTreeCuda3dIndexParams(leaf_size));
    index.buildIndex();

    std::vector<float> qpts;
    fillQueryPoints(pts_data,idx_data,qpts,idx_num);
    flann::Matrix<float> query_dataset(qpts.data(),idx_num,3);

    std::vector<std::vector<int> > indices;
    std::vector<std::vector<float> > dists;
    flann::SearchParams search_params;
    search_params.sorted=false;
    index.radiusSearch(query_dataset,indices,dists,radius*radius,search_params);
    list= vec2DToListNDArray2DInt(indices);
    Py_END_ALLOW_THREADS

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

    Py_BEGIN_ALLOW_THREADS
    flann::Matrix<float> dataset(pts_data,pt_num,3);
    flann::Index<flann::L2_Simple<float> > index(dataset,flann::KDTreeSingleIndexParams(leaf_size));
    index.buildIndex();

    std::vector<std::vector<int> > indices;
    std::vector<std::vector<float> > dists;
    flann::SearchParams search_params;
    index.radiusSearch(dataset,indices,dists,radius*radius,search_params);
    list= vec2DToListNDArray2DInt(indices);
    Py_END_ALLOW_THREADS

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

    Py_BEGIN_ALLOW_THREADS
    flann::Matrix<float> dataset(pts_data,pt_num,3);
    flann::KDTreeCuda3dIndex<flann::L2_Simple<float> > index(dataset,flann::KDTreeSingleIndexParams(leaf_size));
    index.buildIndex();

    std::vector<float> qpts;
    fillQueryPoints(pts_data,idx_data,qpts,idx_num);
    flann::Matrix<float> query_dataset(qpts.data(),idx_num,3);

    std::vector<std::vector<int> > indices;
    std::vector<std::vector<float> > dists;
    flann::SearchParams search_params;
    search_params.sorted=false;
    index.radiusSearch(query_dataset,indices,dists,radius*radius,search_params);
    list= vec2DToListNDArray2DInt(indices);
    Py_END_ALLOW_THREADS

    return list;
}

void gridDownSample(
        float* pts,
        int pt_num,
        int pt_stride,
        float sample_stride,
        std::vector<int>& ds_idxs
);

void gridDownSampleV2(
        float* pts,
        int pt_num,
        int pt_stride,
        float sample_stride,
        std::vector<int>& ds_idxs,
        std::vector<int>& ds_gidxs
);

static PyObject*
gridDownsampleGPU(const bpn::ndarray &pts, float sample_stride, bool output_gidxs = false)
{
    assert(pts.shape(1)>=3);
    assert(pts.get_flags()&bpn::ndarray::C_CONTIGUOUS); //must be contiguous
    assert(std::strcmp(bp::extract<const char *>(bp::str(pts.get_dtype())),"float32")==0);

    float* pts_data=reinterpret_cast<float*>(pts.get_data());
    int pt_num=pts.shape(0),pt_stride=pts.shape(1);
    std::vector<int> ds_idxs;
    PyObject* result;
    Py_BEGIN_ALLOW_THREADS
    if(!output_gidxs)
    {
        gridDownSample(pts_data,pt_num,pt_stride,sample_stride,ds_idxs);
        result=vec1DToNDArray1DInt(ds_idxs);
    }
    else
    {
        std::vector<int> ds_gidxs;
        gridDownSampleV2(pts_data,pt_num,pt_stride,sample_stride,ds_idxs,ds_gidxs);
        PyObject* ds_idxs_arr=vec1DToNDArray1DInt(ds_idxs);
        PyObject* ds_gidxs_arr=vec1DToNDArray1DInt(ds_gidxs);
        bp::list* list=new bp::list();
        list->append(bp::handle<>(ds_idxs_arr));
        list->append(bp::handle<>(ds_gidxs_arr));
        result=list->ptr();
    }
    Py_END_ALLOW_THREADS
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

    Py_BEGIN_ALLOW_THREADS
    std::vector<std::vector<int> > block_idxs=
            sampleRotatedBlockGPUImpl(pts_data, pt_num, pt_stride, sample_stride, block_size, rotate_angle, gpu_index);
    list=vec2DToListNDArray2DInt(block_idxs);
    Py_END_ALLOW_THREADS

    return list;

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
);

PyObject* computeCovarsGPU(
        const bpn::ndarray& ixyz,            // [pn,ps]
        const bpn::ndarray& inidxs,          // [csum]
        const bpn::ndarray& inidxs_lens,     // [pn]
        const bpn::ndarray& inidxs_bgs,      // [pn]
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
    assert(inidxs_lens.shape(0)==pn);
    assert(inidxs_lens.get_flags()&bpn::ndarray::C_CONTIGUOUS);
    assert(std::strcmp(bp::extract<const char *>(bp::str(inidxs_lens.get_dtype())),"int32")==0);

    assert(inidxs_bgs.get_nd()==1);
    assert(inidxs_bgs.shape(0)==pn);
    assert(inidxs_bgs.get_flags()&bpn::ndarray::C_CONTIGUOUS);
    assert(std::strcmp(bp::extract<const char *>(bp::str(inidxs_bgs.get_dtype())),"int32")==0);

    float* xyz_data = reinterpret_cast<float*>(ixyz.get_data());
    int* inidxs_data = reinterpret_cast<int*>(inidxs.get_data());
    int* inidxs_lens_data = reinterpret_cast<int*>(inidxs_lens.get_data());
    int* inidxs_bgs_data = reinterpret_cast<int*>(inidxs_bgs.get_data());

    PyObject* covars=npArray2Df(pn,9);
    Py_BEGIN_ALLOW_THREADS
    float* covars_data= reinterpret_cast<float*>(PyArray_DATA((PyArrayObject*)covars));
    computeCovarsGPUImpl(xyz_data,inidxs_data,inidxs_lens_data,inidxs_bgs_data,covars_data,pn,ps,csum,gpu_index);
    Py_END_ALLOW_THREADS

    return covars;
}

BOOST_PYTHON_FUNCTION_OVERLOADS(findNeighborRadiusAllGPUOverloads, findNeighborRadiusAllGPU, 2, 3);
BOOST_PYTHON_FUNCTION_OVERLOADS(findNeighborRadiusPartGPUOverloads, findNeighborRadiusPartGPU, 3, 4);
BOOST_PYTHON_FUNCTION_OVERLOADS(findNeighborRadiusAllCPUOverloads, findNeighborRadiusAllCPU, 2, 3);
BOOST_PYTHON_FUNCTION_OVERLOADS(findNeighborRadiusPartCPUOverloads, findNeighborRadiusPartCPU, 3, 4);

BOOST_PYTHON_FUNCTION_OVERLOADS(gridDownsampleGPUOverloads, gridDownsampleGPU, 2, 3);
BOOST_PYTHON_FUNCTION_OVERLOADS(sampleRotatedBlockGPUOverloads, sampleRotatedBlockGPU, 4, 5);
BOOST_PYTHON_FUNCTION_OVERLOADS(computeCovarsGPUOverloads, computeCovarsGPU, 4, 5);

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
}
