#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <string>
#include <filesystem>
#include "unistd.h"
#include "python3.12/Python.h"

void execute_python_code(
  PyInterpreterConfig config, const std::string& module_name, const std::string& func_name, std::mutex& print_mutex) {
  const auto currentDir = std::filesystem::current_path();
  // PyGILState_STATE gstate;
  // interpreter 1
  PyThreadState *tstate1 = NULL;
  // gstate = PyGILState_Ensure();
  PyStatus status1 = Py_NewInterpreterFromConfig(&tstate1, &config);
  if (PyStatus_Exception(status1)) {
    std::cout << "Failed\n";
    return;
  }
  PyThreadState_Swap(tstate1);
  PyObject *sysPath = PySys_GetObject("path");
  PyList_Append(sysPath, PyUnicode_FromString(currentDir.c_str()));
  PyObject* myModule = PyImport_ImportModule(module_name.c_str());
  PyErr_Print();
  PyObject* myFunction = PyObject_GetAttrString(myModule, func_name.c_str());
  PyObject* res = PyObject_CallObject(myFunction, NULL);
  // PyGILState_Release(gstate);
  if (res) {
    const int x = (int)PyLong_AsLong(res);
    {
      std::lock_guard<std::mutex> guard(print_mutex);
      std::cerr << "C++ thread id = " << gettid() << ", python result = " << x << std::endl;
    }
  }
  // interpreter 2
  PyThreadState *tstate2 = NULL;
  // gstate = PyGILState_Ensure();
  PyStatus status2 = Py_NewInterpreterFromConfig(&tstate2, &config);
  if (PyStatus_Exception(status2)) {
    std::cout << "Failed\n";
    return;
  }
  PyThreadState_Swap(tstate2);
  sysPath = PySys_GetObject("path");
  PyList_Append(sysPath, PyUnicode_FromString(currentDir.c_str()));
  myModule = PyImport_ImportModule(module_name.c_str());
  PyErr_Print();
  myFunction = PyObject_GetAttrString(myModule, func_name.c_str());
  res = PyObject_CallObject(myFunction, NULL);
  // PyGILState_Release(gstate);
  if (res) {
    const int x = (int)PyLong_AsLong(res);
    {
      std::lock_guard<std::mutex> guard(print_mutex);
      std::cerr << "C++ thread id = " << gettid() << ", python result = " << x << std::endl;
    }
  }
  // destroy interpreter 1
  PyThreadState_Swap(tstate1);
  Py_EndInterpreter(tstate1);
  // destroy interpreter 2
  PyThreadState_Swap(tstate2);
  Py_EndInterpreter(tstate2);
}

int main(int argc, char* argv[]) {
  std::mutex print_mutex;
  std::string pymodule;
  std::string pyfuncname;
  if (argc == 3) {
    pyfuncname = std::string(argv[2]);
    argc--;
  } else {
    pyfuncname = "gettid";
  }
  if (argc == 2) {
    pymodule = std::string(argv[1]);
    argc--;
  } else {
    pymodule = "gettid";
  }
  Py_Initialize();
  PyInterpreterConfig config = {
    .check_multi_interp_extensions = 1,
    .gil = PyInterpreterConfig_OWN_GIL,
  };
  const int num_threads = 4;
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(
      execute_python_code, config, pymodule, pyfuncname, std::ref(print_mutex));
  }
  for (int i = 0; i < num_threads; ++i) {
    threads[i].join();
  }
  int status_exit = Py_FinalizeEx();
  std::cout << "status_exit: " << status_exit << std::endl;
  return 0;
}
