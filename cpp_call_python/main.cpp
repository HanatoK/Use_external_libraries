#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
  // TODO: I don't know if I do Py_DECREF correctly...
  if (argc < 2) {
    return 1;
  }
  Py_Initialize();
  PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;
  pName = PyUnicode_FromString(argv[1]);
  pModule = PyImport_Import(pName);
  Py_DECREF(pName);
  pFunc = PyObject_GetAttrString(pModule, "test");
  pArgs = PyTuple_Pack(1, PyUnicode_FromString("Greg"));
  pValue = PyObject_CallObject(pFunc, pArgs);
  Py_DECREF(pFunc);
  Py_DECREF(pArgs);
  auto result = _PyUnicode_AsString(pValue);
  Py_DECREF(pValue);
  std::cout << result << std::endl;
  // calculate from python
  const std::vector<double> X{1,2,3,4};
  PyObject *pFunc2 = PyObject_GetAttrString(pModule, "calc_xxx");
  if (pFunc2 != NULL) {
    PyObject *listObj = PyList_New(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
      PyObject *num = PyFloat_FromDouble(X[i]);
      PyList_SetItem(listObj, static_cast<Py_ssize_t>(i), num);
    }
    PyObject *pArgs2 = PyTuple_Pack(1, listObj);
    Py_DECREF(listObj);
    PyObject *pValue2 = PyObject_CallObject(pFunc2, pArgs2);
    Py_DECREF(pArgs2);
    std::cout << PyFloat_AsDouble(pValue2) << std::endl;
    Py_DECREF(pValue2);
  }
  Py_DECREF(pModule);
  Py_Finalize();
  return 0;
}
