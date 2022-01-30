#pragma once

#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>

#define USE_DEVICE CL_DEVICE_TYPE_ALL

#define DIM 4
#define MAX_PLATFORMS 10
#define MAX_DEVICES 20

typedef cl_double real;
