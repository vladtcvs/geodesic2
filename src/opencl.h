#pragma once

#include <config.h>

struct calculation_unit_s {
    cl_context context;        // compute context
    cl_program program;        // compute program
    cl_kernel kernel;          // compute kernel
    cl_command_queue queue;   // compute command queue
};

struct opencl_state_s
{
    cl_uint num_platforms;
    cl_platform_id platform_ids[MAX_PLATFORMS];

    cl_uint num_devices[MAX_PLATFORMS];
    cl_device_id device_ids[MAX_PLATFORMS][MAX_DEVICES];

    struct calculation_unit_s units[MAX_PLATFORMS][MAX_DEVICES];
};

const char *opencl_error(int resv);
void init_opencl(struct opencl_state_s *state);
void init_opencl_program(struct opencl_state_s *state, const char *source);
void release_opencl(struct opencl_state_s *state);
