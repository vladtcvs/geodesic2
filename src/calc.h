#pragma once

#include <stdio.h>
#include <opencl.h>

void perform_calculation(struct calculation_unit_s *unit,
                         real T, real h,
                         real *pos,
                         real *dir,
                         cl_int *finished,
                         size_t num_objects,
                         real *args,
                         size_t num_args,
                         FILE **output,
                         cl_int num_steps);
