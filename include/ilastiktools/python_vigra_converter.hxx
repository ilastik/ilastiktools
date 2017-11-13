#ifndef PYTHON_VIGRA_CONVERTER_
#define PYTHON_VIGRA_CONVERTER_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vigra/multi_array.hxx>
#include <vigra/tinyvector.hxx>
#include <exception>
#include <iostream>

template<typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec)
{
    out << "(";
    for(size_t i = 0; i < vec.size() - 1; i++)
    {
        out << vec[i] << ", ";
    }
    if(vec.size() > 0)
        out << vec.back();
    out << ")";
    return out;
}

template<int DIM, typename DTYPE>
vigra::MultiArrayView<DIM, DTYPE> numpy_to_vigra(pybind11::array_t<DTYPE, pybind11::array::f_style | pybind11::array::forcecast> py_array)
{
    pybind11::buffer_info info = py_array.request();
    /**
    struct buffer_info {
        void *ptr;
        size_t itemsize;
        std::string format;
        int ndim;
        std::vector<size_t> shape;
        std::vector<size_t> strides;
    };
    */

    if(info.ndim != DIM)
        throw std::runtime_error("Dimension mismatch between function and argument");

    vigra::TinyVector<int64_t, DIM> shape;
    vigra::TinyVector<int64_t, DIM> strides;

    for(int i = 0; i < DIM; i++)
    {
        shape[i] = info.shape[i];
        strides[i] = info.strides[i] / sizeof(DTYPE); // vigra uses stride = num elements, pybind11 num bytes
    }

    auto ptr = static_cast<DTYPE *>(info.ptr);
    vigra::MultiArrayView<DIM, DTYPE> out(shape, strides, ptr);
    return out;
}

template<int DIM, typename DTYPE>
pybind11::array_t<DTYPE, pybind11::array::f_style | pybind11::array::forcecast> vigra_to_numpy(vigra::MultiArrayView<DIM, DTYPE> vigra_array)
{
    std::vector<size_t> strides;
    std::vector<size_t> shape;

    for(int i = 0; i < DIM; i++)
    {
        strides.push_back(vigra_array.stride()[i] * sizeof(DTYPE));
        shape.push_back(vigra_array.shape()[i]);
    }

    return pybind11::array(pybind11::buffer_info(vigra_array.data(), sizeof(DTYPE),
                     pybind11::format_descriptor<DTYPE>::value,
                     DIM, shape, strides));
}

template<int DIM, typename DTYPE>
const vigra::TinyVector<DTYPE, DIM> numpy_to_tiny_vector(pybind11::array_t<DTYPE, pybind11::array::f_style | pybind11::array::forcecast> py_array)
{
    pybind11::buffer_info info = py_array.request();

    if(info.ndim != 1)
        throw std::runtime_error("Can only convert vectors!");
    if(info.shape[0] != DIM)
        throw std::runtime_error(std::string("Wrong number of elements in vector! Expected ") + std::to_string(DIM)
                                 + std::string(" got ") + std::to_string(info.shape[0]));

    auto ptr = static_cast<DTYPE *>(info.ptr);
    size_t stride = info.strides[0] / sizeof(DTYPE);

    vigra::TinyVector<DTYPE, DIM> out;

    for(int i = 0; i < DIM; i++)
    {
        out[i] = *ptr;
        ptr += stride;
    }

    return out;
}

template<int DIM, typename DTYPE>
pybind11::array_t<DTYPE, pybind11::array::f_style | pybind11::array::forcecast> tiny_vector_to_numpy(vigra::TinyVector<DTYPE, DIM> tiny_vec)
{
    std::vector<size_t> strides = { sizeof(DTYPE) };
    std::vector<size_t> shape = { DIM };

    return pybind11::array(pybind11::buffer_info(static_cast<void*>(tiny_vec.data()), sizeof(DTYPE),
                     pybind11::format_descriptor<DTYPE>::value,
                     1, shape, strides));
}

#endif // PYTHON_VIGRA_CONVERTER_