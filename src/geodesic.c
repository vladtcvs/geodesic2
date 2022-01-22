#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#define SQR(x) ((x) * (x))

#define DIM 4
#define NUM_ARGS 1

#define MAX_PLATFORMS 10

struct opencl_state_s
{
	cl_platform_id platform_ids[MAX_PLATFORMS];
	cl_uint num_platforms;
	cl_device_id device_id;	   // compute device id
	cl_context context;		   // compute context
	cl_command_queue commands; // compute command queue
	cl_program program;		   // compute program
	cl_kernel kernel;		   // compute kernel
};

void init_opencl(struct opencl_state_s *state, const char *source)
{
	int err = clGetPlatformIDs(MAX_PLATFORMS, state->platform_ids, &state->num_platforms);
	err = clGetDeviceIDs(state->platform_ids[0], CL_DEVICE_TYPE_ALL, 1, &state->device_id, NULL);
	state->context = clCreateContext(0, 1, &state->device_id, NULL, NULL, &err);
	state->commands = clCreateCommandQueue(state->context, state->device_id, 0, &err);
	state->program = clCreateProgramWithSource(state->context, 1, (const char **)&source, NULL, &err);
	err = clBuildProgram(state->program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		size_t len;
		char buffer[20480];

		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(state->program, state->device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}
	state->kernel = clCreateKernel(state->program, "kernel_geodesic", &err);
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

void init_ray(double rs, double r, double angle, cl_double *pos, cl_double *dir)
{
	double c = cos(angle);
	double s = sin(angle);

	double k = 1 - rs / r;
	double q = 1 / k * SQR(c) + SQR(s);
	double d = sqrt(k / q);

	double dr = -c * d;
	double df = s * d / r;

	pos[0] = 0;
	pos[1] = r;
	pos[2] = CL_M_PI_2;
	pos[3] = 0;

	dir[0] = -1; // reverse ray tracing
	dir[1] = dr;
	dir[2] = 0;
	dir[3] = df;
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

int main(void)
{
	int i;

	struct opencl_state_s opencl_state;

	// Black hole and metric
	cl_double rs = 1;
	cl_double args[NUM_ARGS] = {rs};

	// Simulation
	cl_double T = 150;
	cl_double h = 5e-4;
	cl_int num_steps = 100;

	// Init viewer light
	// load it from file

	FILE *input = fopen("input.csv", "rt");
	
	int num_objects = file_lines(input) - 1;

	cl_double *pos   = malloc(sizeof(cl_double) * DIM * num_objects);
	cl_double *dir   = malloc(sizeof(cl_double) * DIM * num_objects);
    cl_bool   *finished = malloc(sizeof(cl_bool) * num_objects);

	fseek(input, 0, SEEK_SET);
	char line[1024];
	fgets(line, 1024, input);
	for (i = 0; i < num_objects; i++)
	{
		fgets(line, 1024, input);
		double cpos[4];
		double cdir[4];
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
		finished[i] = false;

	size_t global;
	size_t local;

	cl_mem pos_mem;
	cl_mem dir_mem;
    cl_mem finished_mem;
	cl_mem args_mem;

	const char *source_fname = BINROOT "/geodesic.cl";
	const char *source = load_source(source_fname);
	init_opencl(&opencl_state, source);

	pos_mem = clCreateBuffer(opencl_state.context, CL_MEM_READ_WRITE, sizeof(cl_double) * num_objects * DIM, NULL, NULL);
	dir_mem = clCreateBuffer(opencl_state.context, CL_MEM_READ_WRITE, sizeof(cl_double) * num_objects * DIM, NULL, NULL);
    finished_mem = clCreateBuffer(opencl_state.context, CL_MEM_READ_WRITE, sizeof(cl_bool) * num_objects, NULL, NULL);
	args_mem = clCreateBuffer(opencl_state.context, CL_MEM_READ_WRITE, sizeof(cl_double) * NUM_ARGS, NULL, NULL);

	// Write initial positions and directions
    int err;
	err = clEnqueueWriteBuffer(opencl_state.commands, pos_mem, CL_TRUE, 0, sizeof(cl_double) * num_objects * DIM, pos, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(opencl_state.commands, dir_mem, CL_TRUE, 0, sizeof(cl_double) * num_objects * DIM, dir, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(opencl_state.commands, finished_mem, CL_TRUE, 0, sizeof(cl_bool) * num_objects, finished, 0, NULL, NULL);

	// Write arguments
	err = clEnqueueWriteBuffer(opencl_state.commands, args_mem, CL_TRUE, 0, sizeof(cl_double) * NUM_ARGS, args, 0, NULL, NULL);

	err = clSetKernelArg(opencl_state.kernel, 0, sizeof(cl_int), &num_steps);
	err |= clSetKernelArg(opencl_state.kernel, 1, sizeof(cl_mem), &pos_mem);
	err |= clSetKernelArg(opencl_state.kernel, 2, sizeof(cl_mem), &dir_mem);
    err |= clSetKernelArg(opencl_state.kernel, 3, sizeof(cl_mem), &finished_mem);
	err |= clSetKernelArg(opencl_state.kernel, 4, sizeof(cl_double), &h);
	err |= clSetKernelArg(opencl_state.kernel, 5, sizeof(cl_mem), &args_mem);

	// Prepare calculation group
	err = clGetKernelWorkGroupInfo(opencl_state.kernel, opencl_state.device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
	global = num_objects;
	if (local > global)
		local = global;

	// Run simulation
	cl_double t;
	for (t = 0; t < T; t += h*num_steps)
	{
		err = clEnqueueNDRangeKernel(opencl_state.commands, opencl_state.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
		clFinish(opencl_state.commands);

		err = clEnqueueReadBuffer(opencl_state.commands, pos_mem, CL_TRUE, 0, sizeof(cl_double) * num_objects * DIM, pos, 0, NULL, NULL);
		err = clEnqueueReadBuffer(opencl_state.commands, dir_mem, CL_TRUE, 0, sizeof(cl_double) * num_objects * DIM, dir, 0, NULL, NULL);
        err = clEnqueueReadBuffer(opencl_state.commands, finished_mem, CL_TRUE, 0, sizeof(cl_bool) * num_objects, finished, 0, NULL, NULL);

        /*if (!finished[0])
        {
            printf("%lf, %lf, %lf, %lf, %lf, %i\n", t, pos[0], pos[1], pos[2], pos[3], finished[0]);
        }*/

		bool all_collided = true;
		for (i = 0; i < num_objects; i++)
		{
			if (!finished[i])
            {
                all_collided = false;
				break;
            }
		}

		if (all_collided)
			break;
	}
	clFinish(opencl_state.commands);

	clReleaseMemObject(pos_mem);
	clReleaseMemObject(dir_mem);
    clReleaseMemObject(finished_mem);
	clReleaseMemObject(args_mem);

	release_opencl(&opencl_state);
	
	FILE *output = fopen("output.csv", "wt");
	fprintf(output, "collided");
	for (i = 0; i < DIM; i++)
		fprintf(output, ", pos_%i", i);
	for (i = 0; i < DIM; i++)
		fprintf(output, ", dir_%i", i);
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
