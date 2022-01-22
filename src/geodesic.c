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

struct opencl_state_s
{
    cl_uint num_platforms;
	cl_platform_id platform_ids[MAX_PLATFORMS];
	
    cl_uint num_devices;
	cl_device_id device_ids[MAX_PLATFORMS * MAX_DEVICES];

	cl_context context;		                            // compute context
	cl_command_queue commands;                          // compute command queue
	cl_program program;                                 // compute program
	cl_kernel kernel;                                   // compute kernel
};

void init_opencl(struct opencl_state_s *state, const char *source)
{
    int i;
	int err = clGetPlatformIDs(MAX_PLATFORMS, state->platform_ids, &state->num_platforms);

    state->num_devices = 0;
    for (i = 0; i < state->num_platforms && i < MAX_PLATFORMS; i++)
    {
        cl_uint nd;
	    err = clGetDeviceIDs(state->platform_ids[i], CL_DEVICE_TYPE_ALL, MAX_DEVICES, &state->device_ids[state->num_devices], &nd);
        state->num_devices += nd;
    }
	state->context = clCreateContext(0, state->num_devices, state->device_ids, NULL, NULL, &err);

	state->program = clCreateProgramWithSource(state->context, 1, (const char **)&source, NULL, &err);
	err = clBuildProgram(state->program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		size_t len;
		char buffer[20480];

		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(state->program, state->device_ids[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}
	state->kernel = clCreateKernel(state->program, "kernel_geodesic", &err);

    state->commands = clCreateCommandQueue(state->context, state->device_ids[0], 0, &err);
}

void release_opencl(struct opencl_state_s *state)
{
	clReleaseProgram(state->program);
	clReleaseKernel(state->kernel);
	clReleaseCommandQueue(state->commands);
	clReleaseContext(state->context);
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
    sscanf(argv[7], "%i",  &num_steps);

	// Init viewer light
	// load it from file

    const char *input_fname = argv[1];
    const char *output_fname = argv[2];
    const char *metric_fname = argv[3];
    const char *args_fname = argv[4];

    /* Read arguments */
    FILE *af = fopen(args_fname, "rt");
    int num_args = file_lines(af) - 1;

    cl_double *args = malloc(sizeof(cl_double) * num_args);
    
    fseek(af, 0, SEEK_SET);
	fgets(line, 1024, af);
    for (i = 0; i < num_args; i++)
	{
        fgets(line, 1024, af);
        sscanf(line, "%lf", &args[i]);
    }
    fclose(af);
    
    /* Read initial state */
	FILE *input = fopen(input_fname, "rt");
	int num_objects = file_lines(input) - 1;

	cl_double *pos      = malloc(sizeof(cl_double) * DIM * num_objects);
	cl_double *dir      = malloc(sizeof(cl_double) * DIM * num_objects);
    cl_int    *finished = malloc(sizeof(cl_int) * num_objects);
	
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
			pos[i*DIM+j] = cpos[j];
			dir[i*DIM+j] = cdir[j];
		}
	}
	fclose(input);

	printf("Loaded %i objects\n", num_objects);

	for (i = 0; i < num_objects; i++)
		finished[i] = 0;

	size_t global;
	size_t local;

	cl_mem pos_mem;
	cl_mem dir_mem;
    cl_mem finished_mem;
	cl_mem args_mem;

	const char *source_fname = BINROOT "/geodesic.cl";
	const char *source = load_source(source_fname);
    const char *metric = load_source(metric_fname);
    char *kernel = malloc(strlen(source) + strlen(metric) + 3);
    strcpy(kernel, source);
    strcat(kernel, metric);

	init_opencl(&opencl_state, kernel);

	pos_mem = clCreateBuffer(opencl_state.context, CL_MEM_READ_WRITE, sizeof(cl_double) * num_objects * DIM, NULL, NULL);
	dir_mem = clCreateBuffer(opencl_state.context, CL_MEM_READ_WRITE, sizeof(cl_double) * num_objects * DIM, NULL, NULL);
    finished_mem = clCreateBuffer(opencl_state.context, CL_MEM_READ_WRITE, sizeof(cl_int) * num_objects, NULL, NULL);
	args_mem = clCreateBuffer(opencl_state.context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, sizeof(cl_double) * num_args, NULL, NULL);

	// Write initial positions and directions
    int err;
	err = clEnqueueWriteBuffer(opencl_state.commands, pos_mem, CL_TRUE, 0, sizeof(cl_double) * num_objects * DIM, pos, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(opencl_state.commands, dir_mem, CL_TRUE, 0, sizeof(cl_double) * num_objects * DIM, dir, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(opencl_state.commands, finished_mem, CL_TRUE, 0, sizeof(cl_int) * num_objects, finished, 0, NULL, NULL);

	// Write arguments
	err = clEnqueueWriteBuffer(opencl_state.commands, args_mem, CL_TRUE, 0, sizeof(cl_double) * num_args, args, 0, NULL, NULL);

	err = clSetKernelArg(opencl_state.kernel, 0, sizeof(cl_int), &num_steps);
	err |= clSetKernelArg(opencl_state.kernel, 1, sizeof(cl_mem), &pos_mem);
	err |= clSetKernelArg(opencl_state.kernel, 2, sizeof(cl_mem), &dir_mem);
    err |= clSetKernelArg(opencl_state.kernel, 3, sizeof(cl_mem), &finished_mem);
	err |= clSetKernelArg(opencl_state.kernel, 4, sizeof(cl_double), &h);
	err |= clSetKernelArg(opencl_state.kernel, 5, sizeof(cl_mem), &args_mem);

	// Prepare calculation group
	err = clGetKernelWorkGroupInfo(opencl_state.kernel, opencl_state.device_ids[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
	global = num_objects;
	if (local > global)
		local = global;

	// Run simulation
	cl_double t;
	for (t = 0; t < T; t += h*num_steps)
	{
		err = clEnqueueNDRangeKernel(opencl_state.commands, opencl_state.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
		clFinish(opencl_state.commands);

		err = clEnqueueReadBuffer(opencl_state.commands, finished_mem, CL_TRUE, 0, sizeof(cl_int) * num_objects, finished, 0, NULL, NULL);
        clFinish(opencl_state.commands);

		bool all_collided = true;
		for (i = 0; i < num_objects; i++)
		{
			if (finished[i] == 0)
            {
                all_collided = false;
				break;
            }
		}

        printf("%lf / %lf\n", t, T);

		if (all_collided)
			break;
	}

    err = clEnqueueReadBuffer(opencl_state.commands, pos_mem, CL_TRUE, 0, sizeof(cl_double) * num_objects * DIM, pos, 0, NULL, NULL);
	err = clEnqueueReadBuffer(opencl_state.commands, dir_mem, CL_TRUE, 0, sizeof(cl_double) * num_objects * DIM, dir, 0, NULL, NULL);    
    err = clEnqueueReadBuffer(opencl_state.commands, finished_mem, CL_TRUE, 0, sizeof(cl_int) * num_objects, finished, 0, NULL, NULL);
	clFinish(opencl_state.commands);

	clReleaseMemObject(pos_mem);
	clReleaseMemObject(dir_mem);
    clReleaseMemObject(finished_mem);
	clReleaseMemObject(args_mem);

	release_opencl(&opencl_state);

	FILE *output = fopen(output_fname, "wt");
	fprintf(output, "collided");
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
			fprintf(output, ", %lf", pos[DIM*i+j]);
		for (j = 0; j < DIM; j++)
			fprintf(output, ", %lf", dir[DIM*i+j]);
		fprintf(output, "\n");
	}

	fclose(output);

	return 0;
}
