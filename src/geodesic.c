#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#define SQR(x) ((x) * (x))

#define DIM 4

#define MAX_PLATFORMS 10
#define MAX_DEVICES 20

#define USE_DEVICE CL_DEVICE_TYPE_ALL

typedef cl_double real;

struct opencl_state_s
{
    cl_uint num_platforms;
    cl_platform_id platform_ids[MAX_PLATFORMS];

    cl_uint num_devices[MAX_PLATFORMS];
    cl_device_id device_ids[MAX_PLATFORMS][MAX_DEVICES];

    struct {
        cl_context context;        // compute context
        cl_program program;        // compute program
        cl_kernel kernel;          // compute kernel
        cl_command_queue commands[MAX_DEVICES]; // compute command queue
    } units[MAX_PLATFORMS];
};

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

void init_opencl_program(struct opencl_state_s *state, const char *source, int platform_id)
{
    if (state->num_devices[platform_id] == 0)
        return;

    int err;
    state->units[platform_id].context = clCreateContext(0, state->num_devices[platform_id], state->device_ids[platform_id], NULL, NULL, &err);

    state->units[platform_id].program = clCreateProgramWithSource(state->units[platform_id].context, 1, (const char **)&source, NULL, &err);
    err = clBuildProgram(state->units[platform_id].program, 0, NULL, NULL, NULL, NULL);

    int i;
    for (i = 0; i < state->num_devices[platform_id]; i++)
    {
        size_t len;
        char buffer[20480];
        int resv = clGetProgramBuildInfo(state->units[platform_id].program, state->device_ids[platform_id][i], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);

        printf("Get info: %s\n", opencl_error(resv));
        printf("Log: %s\n", buffer);
    }

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to build program executable!\n");
        exit(1);
    }

    state->units[platform_id].kernel = clCreateKernel(state->units[platform_id].program, "kernel_geodesic", &err);

    for (i = 0; i < state->num_devices[platform_id]; i++)
        state->units[platform_id].commands[i] = clCreateCommandQueue(state->units[platform_id].context, state->device_ids[platform_id][i], 0, &err);
}

void release_opencl(struct opencl_state_s *state)
{
    int i, j;

    for (i = 0; i < state->num_platforms; i++)
    {
        if (state->num_devices[i] == 0)
            continue;
        clReleaseProgram(state->units[i].program);
        clReleaseKernel(state->units[i].kernel);
        clReleaseContext(state->units[i].context);
        for (j = 0; j < state->num_devices[i]; j++)
            clReleaseCommandQueue(state->units[i].commands[j]);
    }
}

const char *load_source(const char *fname)
{
    FILE *f = fopen(fname, "rb");
    if (f == NULL)
    {
        printf("Can not open file %s. Exiting.", fname);
        exit(1);
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *source = malloc(fsize + 1);
    fread(source, 1, fsize, f);
    fclose(f);
    source[fsize] = 0;

    return source;
}

int file_lines(FILE *f)
{
    int num_lines = 0;
    fseek(f, 0, SEEK_SET);
    while (!feof(f))
    {
        char c = fgetc(f);
        if (c == '\n')
            num_lines++;
    }
    return num_lines;
}

void perform_calculation(cl_context context,
                         cl_command_queue queue,
                         cl_kernel kernel,
                         real T, real h,
                         real *pos,
                         real *dir,
                         cl_int *finished,
                         size_t num_objects,
                         real *args,
                         size_t num_args,
                         FILE **output,
                         cl_int num_steps)
{
    int err;
    int i;

    cl_mem pos_mem;
    cl_mem dir_mem;
    cl_mem finished_mem;

    cl_mem args_mem;

    pos_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(real) * num_objects * DIM, NULL, NULL);
    dir_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(real) * num_objects * DIM, NULL, NULL);
    finished_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int) * num_objects, NULL, NULL);
    args_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, sizeof(real) * num_args, NULL, NULL);

    clEnqueueWriteBuffer(queue, pos_mem, CL_TRUE, 0, sizeof(real) * num_objects * DIM, pos, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, dir_mem, CL_TRUE, 0, sizeof(real) * num_objects * DIM, dir, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, finished_mem, CL_TRUE, 0, sizeof(cl_int) * num_objects, finished, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, args_mem, CL_TRUE, 0, sizeof(real) * num_args, args, 0, NULL, NULL);

    clFinish(queue);

    clSetKernelArg(kernel, 0, sizeof(cl_int), &num_steps);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &pos_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &dir_mem);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &finished_mem);
    clSetKernelArg(kernel, 4, sizeof(real), &h);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &args_mem);

    real t;

    if (output)
    {
        for (i = 0; i < num_objects; i++)
        {
           if (finished[i])
               fprintf(output[i], "true");
           else
               fprintf(output[i], "false");
           fprintf(output[i], ", %0.12lf", 0.0);
           int j;
           for (j = 0; j < DIM; j++)
               fprintf(output[i], ", %0.12lf", (double)pos[DIM * i + j]);
           for (j = 0; j < DIM; j++)
               fprintf(output[i], ", %0.12lf", (double)dir[DIM * i + j]);
           fprintf(output[i], "\n");
        }
    }

    for (t = 0; t < T; t += h * num_steps)
    {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &num_objects, NULL, 0, NULL, NULL);
        clFinish(queue);

        if (output)
        {
            err = clEnqueueReadBuffer(queue, pos_mem, CL_TRUE, 0, sizeof(real) * DIM * num_objects, pos, 0, NULL, NULL);
            err = clEnqueueReadBuffer(queue, dir_mem, CL_TRUE, 0, sizeof(real) * DIM * num_objects, dir, 0, NULL, NULL);
        }
        err = clEnqueueReadBuffer(queue, finished_mem, CL_TRUE, 0, sizeof(cl_int) * num_objects, finished, 0, NULL, NULL);
        clFinish(queue);

        bool all_collided = true;
        for (i = 0; i < num_objects; i++)
        {
            if (finished[i] == 0)
            {
                all_collided = false;
                break;
            }
        }

        if (output)
        {
            for (i = 0; i < num_objects; i++)
            {
                if (finished[i])
                    fprintf(output[i], "true");
                else
                    fprintf(output[i], "false");
                fprintf(output[i], ", %0.12lf", t);
                int j;
                for (j = 0; j < DIM; j++)
                    fprintf(output[i], ", %0.12lf", (double)pos[DIM * i + j]);
                for (j = 0; j < DIM; j++)
                    fprintf(output[i], ", %0.12lf", (double)dir[DIM * i + j]);
                fprintf(output[i], "\n");
                fflush(output[i]);
            }
        }

        printf("%lf / %lf\n", t, T);

        if (all_collided)
            break;
    }

    err = clEnqueueReadBuffer(queue, pos_mem, CL_TRUE, 0, sizeof(real) * num_objects * DIM, pos, 0, NULL, NULL);
    err = clEnqueueReadBuffer(queue, dir_mem, CL_TRUE, 0, sizeof(real) * num_objects * DIM, dir, 0, NULL, NULL);
    err = clEnqueueReadBuffer(queue, finished_mem, CL_TRUE, 0, sizeof(cl_int) * num_objects, finished, 0, NULL, NULL);
    clFinish(queue);

    clReleaseMemObject(pos_mem);
    clReleaseMemObject(dir_mem);
    clReleaseMemObject(finished_mem);
    clReleaseMemObject(args_mem);
}

int main(int argc, const char **argv)
{
    int i;

    if (argc < 8)
    {
        printf("Usage: geodesic2 input.csv output.csv metric.cl args.csv <T> <h> <num steps>\n");
        return 1;
    }

    char line[1024];
    struct opencl_state_s opencl_state;

    // Simulation
    double T;
    double h;
    int num_steps;

    sscanf(argv[5], "%lf", &T);
    sscanf(argv[6], "%lf", &h);
    sscanf(argv[7], "%i", &num_steps);

    // Init viewer light
    // load it from file

    const char *input_fname = argv[1];
    const char *output_fname = argv[2];
    const char *metric_fname = argv[3];
    const char *args_fname = argv[4];

    const char *out_dirname = NULL;
    if (argc >= 9)
    {
        out_dirname = argv[8];
    }

    /* Read arguments */
    FILE *af = fopen(args_fname, "rt");
    int num_args = file_lines(af) - 1;

    real *args = malloc(sizeof(real) * num_args);

    fseek(af, 0, SEEK_SET);
    fgets(line, 1024, af);
    for (i = 0; i < num_args; i++)
    {
        double v;
        fgets(line, 1024, af);
        sscanf(line, "%lf", &v);
        args[i] = v;
    }
    fclose(af);

    /* Read initial state */
    FILE *input = fopen(input_fname, "rt");
    int num_objects = file_lines(input) - 1;

    real *pos = malloc(sizeof(real) * DIM * num_objects);
    real *dir = malloc(sizeof(real) * DIM * num_objects);
    cl_int *finished = malloc(sizeof(cl_int) * num_objects);

    fseek(input, 0, SEEK_SET);
    fgets(line, 1024, input);
    for (i = 0; i < num_objects; i++)
    {
        fgets(line, 1024, input);
        double cpos[DIM];
        double cdir[DIM];
        sscanf(line, "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf", &cpos[0], &cpos[1], &cpos[2], &cpos[3], &cdir[0], &cdir[1], &cdir[2], &cdir[3]);
        int j;
        for (j = 0; j < DIM; j++)
        {
            pos[i * DIM + j] = cpos[j];
            dir[i * DIM + j] = cdir[j];
        }
    }
    fclose(input);

    printf("Loaded %i objects\n", num_objects);

    for (i = 0; i < num_objects; i++)
        finished[i] = 0;

    size_t global;
    size_t local;

    const char *source_fname = BINROOT "/geodesic.cl";
    const char *source = load_source(source_fname);
    const char *metric = load_source(metric_fname);
    char *kernel_source = malloc(strlen(source) + strlen(metric) + 3);
    strcpy(kernel_source, source);
    strcat(kernel_source, metric);

    init_opencl(&opencl_state);

    for (i = 0; i < opencl_state.num_platforms; i++)
        init_opencl_program(&opencl_state, kernel_source, i);

    int platform_id = -1;
    int device_id = -1;

    for (i = 0; i < opencl_state.num_platforms; i++)
    {
        if (opencl_state.num_devices[i] > 0)
        {
            platform_id = i;
            device_id = 0;
            break;
        }
    }

    if (platform_id == -1)
    {
        printf("Can not find OpenCL device! Exiting\n");
        return 0;
    }

    printf("Select platform %i, device %i\n", platform_id, device_id);

    FILE **output_rays = NULL;
    if (out_dirname != NULL)
    {
        output_rays = malloc(sizeof(FILE*)*num_objects);
        for (i = 0; i < num_objects; i++)
        {
            char fname[4096];
            snprintf(fname, 4096, "%s/%05i.csv", out_dirname, i);
            output_rays[i] = fopen(fname, "wt");
            int j;

            /*fprintf(output_rays[i], "collided,t");
            for (j = 0; j < DIM; j++)
                fprintf(output_rays[i], ",pos%i", j);
            for (j = 0; j < DIM; j++)
                fprintf(output_rays[i], ",dir%i", j);
            fprintf(output_rays[i], "\n");*/
        }
    }

    perform_calculation(opencl_state.units[platform_id].context,
                        opencl_state.units[platform_id].commands[device_id],
                        opencl_state.units[platform_id].kernel,
                        T, h, pos, dir, finished, num_objects, args, num_args, output_rays, num_steps);

    release_opencl(&opencl_state);

    FILE *output = fopen(output_fname, "wt");
    fprintf(output, "finished");
    for (i = 0; i < DIM; i++)
        fprintf(output, ",pos%i", i);
    for (i = 0; i < DIM; i++)
        fprintf(output, ",dir%i", i);
    fprintf(output, "\n");

    for (i = 0; i < num_objects; i++)
    {
        if (finished[i])
            fprintf(output, "true");
        else
            fprintf(output, "false");
        int j;
        for (j = 0; j < DIM; j++)
            fprintf(output, ",%lf", (double)pos[DIM * i + j]);
        for (j = 0; j < DIM; j++)
            fprintf(output, ",%lf", (double)dir[DIM * i + j]);
        fprintf(output, "\n");
    }

    if (output_rays != NULL)
    {
        for (i = 0; i < num_objects; i++)
            fclose(output_rays[i]);
    }
    fclose(output);
    return 0;
}
