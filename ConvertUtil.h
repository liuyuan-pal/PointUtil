//
// Created by pal on 18-2-1.
//

#ifndef POINTUTIL_CONVERTUTIL_H
#define POINTUTIL_CONVERTUTIL_H

#include <numpy/ndarrayobject.h>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <vector>
#include <iostream>

inline PyObject* vec2DToListNDArray2DInt(const std::vector<std::vector<int> > &vec)
{
    namespace bp=boost::python;
    namespace bpn=boost::python::numpy;
    bp::list* pylistlist = new bp::list();
    for(int i=0;i<vec.size();i++)
    {
        npy_intp shape[1]={vec[i].size()};
        PyObject* array=PyArray_SimpleNew(1,shape,NPY_INT32);
        void * arr_data = PyArray_DATA((PyArrayObject*)array);
        memcpy(arr_data, &vec[i][0], shape[0] * sizeof(int));
        pylistlist->append(bp::handle<>(array));
    }
    return pylistlist->ptr();
}



inline PyObject* vec1DToNDArray1DInt(std::vector<int>& vec)
{
    namespace bp=boost::python;
    namespace bpn=boost::python::numpy;
    npy_intp shape[1]={vec.size()};
    PyObject* array=PyArray_SimpleNew(1,shape,NPY_INT32);
    void * arr_data = PyArray_DATA((PyArrayObject*)array);
    memcpy(arr_data,vec.data(),shape[0]*sizeof(float));
    return array;
}

inline PyObject* vec1DToNDArray2D(std::vector<float>& vec, unsigned int rows, unsigned int cols)
{
    namespace bp=boost::python;
    namespace bpn=boost::python::numpy;
    npy_intp shape[2]={rows,cols};
    PyObject* array=PyArray_SimpleNew(2,shape,NPY_FLOAT32);
    void * arr_data = PyArray_DATA((PyArrayObject*)array);
    memcpy(arr_data,vec.data(),rows*cols*sizeof(float));
    return array;
}


inline PyObject* vec1DToNDArray3D(std::vector<float>& vec, unsigned int dim1, unsigned int dim2, unsigned int dim3)
{
    namespace bp=boost::python;
    namespace bpn=boost::python::numpy;
    npy_intp shape[3]={dim1,dim2,dim3};
    PyObject* array=PyArray_SimpleNew(3,shape,NPY_FLOAT32);
    void * arr_data = PyArray_DATA((PyArrayObject*)array);
    memcpy(arr_data,vec.data(),vec.size()*sizeof(float));
    return array;
}

inline PyObject* npArray2Df(int dim1,int dim2)
{
    npy_intp shape[2]={dim1,dim2};
    PyObject* array=PyArray_SimpleNew(2,shape,NPY_FLOAT32);
    return array;
};

#endif //POINTUTIL_CONVERTUTIL_H
