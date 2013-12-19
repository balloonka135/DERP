#ifndef PYTHON_BINDINGS_H_
#define PYTHON_BINDINGS_H_

#include <Python.h>
#include "asmmodel.h"

typedef struct {
    PyObject_HEAD
    StatModel::ASMModel* model;
} AsmModelObject;

extern "C" {
    PyObject* LoadAsmModel(PyObject* self, PyObject* args);
    PyObject* FitImage(AsmModelObject* self, PyObject* image);

    int AsmModelInit(AsmModelObject* self, PyObject* args, PyObject* kwds);
    PyObject* AsmModelNew(PyTypeObject* type, PyObject* args, PyObject* kwds);
    void AmsModelDealloc(AsmModelObject* self);

    void init_arrays();
}

static PyMethodDef AsmModelMethods[] = {
    {"fit_one", (PyCFunction)FitImage, METH_O, "Fit face bounding box image to shape model"},
    {NULL} 
};

static PyTypeObject AsmModelType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "pyams.AsmModel",          /*tp_name*/
    sizeof(AsmModelObject),    /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)AmsModelDealloc,/*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                            /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,           /*tp_flags*/
    "Active shape model wrapper", /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    AsmModelMethods,           /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)AsmModelInit,    /* tp_init */
    0,                         /* tp_alloc */
    AsmModelNew,               /* tp_new */
};

#endif