#include <stdio.h>

#include <config.h>

#include <opencl.h>

const char *opencl_error(int resv)
{
    const char *res;
    switch (resv)
    {
    case 0:
        return "ok";
    case CL_INVALID_DEVICE:
        return "invalid device";
    case CL_INVALID_VALUE:
        return "invalid value";
    case CL_INVALID_PROGRAM:
        return "invalid program";
    case CL_OUT_OF_RESOURCES:
        return "out of resources";
    case CL_OUT_OF_HOST_MEMORY:
        return "out of host memory";
    default:
        return "unknown";
    }
}

void init_opencl(struct opencl_state_s *state)
{
    int i;
    int err = clGetPlatformIDs(MAX_PLATFORMS, state->platform_ids, &state->num_platforms);

    printf("Number of platforms: %i\n", (int)state->num_platforms);
    for (i = 0; i < state->num_platforms && i < MAX_PLATFORMS; i++)
    {
        err = clGetDeviceIDs(state->platform_ids[i], USE_DEVICE, MAX_DEVICES, state->device_ids[i], &state->num_devices[i]);
        printf("Platform %i. Number of devices: %i\n", i, (int)state->num_devices[i]);
    }
}

void init_opencl_program(struct opencl_state_s *state, const char *source)
{
    int pid, did;

    for (pid = 0; pid < state->num_platforms; pid++)
    {
        printf("platform %i\n", pid);
        for (did = 0; did < state->num_devices[pid]; did++)
        {
            printf("\tdevice %i\n", did);
            struct calculation_unit_s *unit = &state->units[pid][did];
            cl_device_id device_id = state->device_ids[pid][did];
            int err;
            unit->context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
            unit->program = clCreateProgramWithSource(unit->context, 1, (const char **)&source, NULL, &err);
            err = clBuildProgram(unit->program, 0, NULL, NULL, NULL, NULL);

            size_t len;
            char buffer[20480];
            int resv = clGetProgramBuildInfo(unit->program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);

            printf("Get info: %s\n", opencl_error(resv));
            printf("Log: %s\n", buffer);

            if (err != CL_SUCCESS)
            {
                printf("Error: Failed to build program executable!\n");
                exit(1);
            }

            unit->kernel = clCreateKernel(unit->program, "kernel_geodesic", &err);
            unit->queue = clCreateCommandQueue(unit->context, device_id, 0, &err);

            unit->max_parallel_points = 1024;
        }
    }
}

void release_opencl(struct opencl_state_s *state)
{
    int i, j;

    for (i = 0; i < state->num_platforms; i++)
    {
        for (j = 0; j < state->num_devices[i]; j++)
        {
            clReleaseProgram(state->units[i][j].program);
            clReleaseKernel(state->units[i][j].kernel);
            clReleaseContext(state->units[i][j].context);
            clReleaseCommandQueue(state->units[i][j].queue);
        }
    }
}
