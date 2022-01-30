#include <config.h>
#include <stdio.h>
#include <stdbool.h>
#include <opencl.h>
#include <calc.h>

void perform_calculation(struct calculation_unit_s *unit,
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

    pos_mem = clCreateBuffer(unit->context, CL_MEM_READ_WRITE, sizeof(real) * num_objects * DIM, NULL, NULL);
    dir_mem = clCreateBuffer(unit->context, CL_MEM_READ_WRITE, sizeof(real) * num_objects * DIM, NULL, NULL);
    finished_mem = clCreateBuffer(unit->context, CL_MEM_READ_WRITE, sizeof(cl_int) * num_objects, NULL, NULL);
    args_mem = clCreateBuffer(unit->context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, sizeof(real) * num_args, NULL, NULL);
 
    clEnqueueWriteBuffer(unit->queue, pos_mem, CL_TRUE, 0, sizeof(real) * num_objects * DIM, pos, 0, NULL, NULL);
    clEnqueueWriteBuffer(unit->queue, dir_mem, CL_TRUE, 0, sizeof(real) * num_objects * DIM, dir, 0, NULL, NULL);
    clEnqueueWriteBuffer(unit->queue, finished_mem, CL_TRUE, 0, sizeof(cl_int) * num_objects, finished, 0, NULL, NULL);
    clEnqueueWriteBuffer(unit->queue, args_mem, CL_TRUE, 0, sizeof(real) * num_args, args, 0, NULL, NULL);
    clFinish(unit->queue);

    clSetKernelArg(unit->kernel, 0, sizeof(cl_int), &num_steps);
    clSetKernelArg(unit->kernel, 1, sizeof(cl_mem), &pos_mem);
    clSetKernelArg(unit->kernel, 2, sizeof(cl_mem), &dir_mem);
    clSetKernelArg(unit->kernel, 3, sizeof(cl_mem), &finished_mem);
    clSetKernelArg(unit->kernel, 4, sizeof(real), &h);
    clSetKernelArg(unit->kernel, 5, sizeof(cl_mem), &args_mem);

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
        err = clEnqueueNDRangeKernel(unit->queue, unit->kernel, 1, NULL, &num_objects, NULL, 0, NULL, NULL);
        clFinish(unit->queue);

        if (output)
        {
            err = clEnqueueReadBuffer(unit->queue, pos_mem, CL_TRUE, 0, sizeof(real) * DIM * num_objects, pos, 0, NULL, NULL);
            err = clEnqueueReadBuffer(unit->queue, dir_mem, CL_TRUE, 0, sizeof(real) * DIM * num_objects, dir, 0, NULL, NULL);
        }
        err = clEnqueueReadBuffer(unit->queue, finished_mem, CL_TRUE, 0, sizeof(cl_int) * num_objects, finished, 0, NULL, NULL);
        clFinish(unit->queue);

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
        {
            printf("All rays collided\n");
            break;
        }
    }

    err = clEnqueueReadBuffer(unit->queue, pos_mem, CL_TRUE, 0, sizeof(real) * num_objects * DIM, pos, 0, NULL, NULL);
    err = clEnqueueReadBuffer(unit->queue, dir_mem, CL_TRUE, 0, sizeof(real) * num_objects * DIM, dir, 0, NULL, NULL);
    err = clEnqueueReadBuffer(unit->queue, finished_mem, CL_TRUE, 0, sizeof(cl_int) * num_objects, finished, 0, NULL, NULL);
    clFinish(unit->queue);

    clReleaseMemObject(pos_mem);
    clReleaseMemObject(dir_mem);
    clReleaseMemObject(finished_mem);
    clReleaseMemObject(args_mem);
}
