#include <Python.h>

#include <cstdio>
#include <iostream>
#include <string>

#include "python_bindings.h"

extern PyObject* PyAsmError;

static PyMethodDef methods[] = {
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initpyasm(void) {
    PyObject *module;

    module = Py_InitModule("pyasm", methods);

    if (module == NULL)
        return;

    PyAsmError = PyErr_NewException("pyasm.error", NULL, NULL);
    Py_INCREF(PyAsmError);
    PyModule_AddObject(module, "error", PyAsmError);

    AsmModelType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&AsmModelType) < 0)
        return;

    Py_INCREF(&AsmModelType);
    PyModule_AddObject(module, "AsmModel", (PyObject *)&AsmModelType);

    init_arrays();
}

int main(int argc, char *argv[]) {
    Py_SetProgramName(argv[0]);

    Py_Initialize();

    initpyasm();

    return 0;
}