#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs, *pValue;

    // Initialize the Python Interpreter
    Py_Initialize();

    // Add the current directory to the Python path
    PyObject *sysPath = PySys_GetObject("path");
    PyObject *curDir = PyUnicode_FromString(".");
    PyList_Append(sysPath, curDir);
    Py_DECREF(curDir);

    // Build the name object
    pName = PyUnicode_DecodeFSDefault("inference");
    // Load the module object
    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        // Initialize the LLM
        pFunc = PyObject_GetAttrString(pModule, "init_llm");
        if (pFunc && PyCallable_Check(pFunc)) {
            pValue = PyObject_CallObject(pFunc, NULL);
            if (pValue != NULL) {
                printf("C Wrapper: %s\n", PyUnicode_AsUTF8(pValue));
                Py_DECREF(pValue);
            }
            Py_DECREF(pFunc);
        } else {
            PyErr_Print();
            fprintf(stderr, "Cannot find function \"init_llm\"\n");
        }

        // Call the generate function
        pFunc = PyObject_GetAttrString(pModule, "generate");
        if (pFunc && PyCallable_Check(pFunc)) {
            const char *prompt = (argc > 1) ? argv[1] : "To be";
            pArgs = PyTuple_New(1);
            pValue = PyUnicode_FromString(prompt);
            PyTuple_SetItem(pArgs, 0, pValue);

            printf("C Wrapper: Generating text for prompt: '%s'...\n", prompt);
            pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);

            if (pValue != NULL) {
                printf("Generated Text:\n----------------\n%s\n----------------\n", PyUnicode_AsUTF8(pValue));
                Py_DECREF(pValue);
            } else {
                PyErr_Print();
                fprintf(stderr, "Call failed\n");
            }
            Py_DECREF(pFunc);
        } else {
            PyErr_Print();
            fprintf(stderr, "Cannot find function \"generate\"\n");
        }
        Py_DECREF(pModule);
    } else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"inference\"\n");
        return 1;
    }

    // Clean up
    Py_Finalize();
    return 0;
}
