#pragma once

#include <config.h>
#include <stdio.h>

struct dispatcher_s {
    real *pos;
    real *dir;
    cl_int *finished;
    FILE **output;
    cl_uint num_objects;
    cl_uint num_completed;
    cl_uint max_per_block;
};

void dispatcher_init(struct dispatcher_s *dispatcher, real *pos, real *dir, cl_int *finished, FILE **output, size_t num_objects);
size_t dispatcher_get_next_block(struct dispatcher_s *dispatcher, real **pos, real **dir, cl_int **finished, FILE ***output);
