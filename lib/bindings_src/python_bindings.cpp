#include <Python.h>
#include <cv.h>

#include <cstddef>
#include <iostream>

#include "conversion.h"
#include "asmmodel.h"
#include "python_bindings.h"

#include "numpy/arrayobject.h"


PyObject* PyAsmError;

static NDArrayConverter array_converter_;

void init_arrays() {
    import_array();
}

PyObject* FitImage(AsmModelObject* self, PyObject* image) {  
    vector< cv::Rect > detected; 
    vector< cv::Point_<int> > point_vec;

    int* cin;

    cv::Mat img = array_converter_.toMat(image); 
    cv::Size imsize = img.size();   
    cv::Rect face(0, 0, imsize.width, imsize.height);
    detected.push_back(face);

    self->model->fitAll(img, detected, 0)[0].toPointList(point_vec);
    size_t points_c = point_vec.size();

    npy_intp dims[] =  { 2*points_c };
    PyArrayObject* vecout = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT32);
    cin = (int*)vecout->data;

    for(size_t i = 0; i < points_c; ++i) {
        cin[i*2] = point_vec[i].x;
        cin[i*2+1] = point_vec[i].y;
    }

    return PyArray_Return(vecout);
}

PyObject* AsmModelNew(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    AsmModelObject* self;

    self = (AsmModelObject*)type->tp_alloc(type, 0);

    return (PyObject*)self;
}

int AsmModelInit(AsmModelObject* self, PyObject* args, PyObject* kwds) {
    const char* buf;   

    if (!PyArg_ParseTuple(args, "s", &buf)) {
        return NULL;
    }

    std::string path(buf);
    self->model = new StatModel::ASMModel(path);

    return 0;
}

void AmsModelDealloc(AsmModelObject* self) {
    delete self->model;
    self->ob_type->tp_free((PyObject*)self);
}