This tool is for calculation of geodesics in black hole metric.

# Geodesics calculator

## Arguments

```
./geodesic input.csv output.csv metric.cl arguments.csv T h num_steps [output_dir/] 
```

## input.csv
file with initial geodesic point and dir: pos0, pos1, pos2, pos3, dir0, dir1, dir2, dir3

## output.csv
file with final geodesic point and dir in the same format

## metric.cl

OpenCL file with several functions, describing metric of spacetime
* `struct tensor_2 metric_tensor(const struct tensor_1 *pos, __global const real *args)` - covariant metric tensor g_{\mu\nu} 
* `bool allowed_area(const struct tensor_1 *pos, __global const real *args)`             - check if specified position is valid
* `bool allowed_delta(const struct tensor_1 *pos, const struct tensor_1 *dir, const struct tensor_1 *dpos, const struct tensor_1 *ddir, __global const real *args)` - check if specified delta is valid
* `struct tensor_2 contravariant_metric_tensor(const struct tensor_2 *g)`                - metric tensor in contravariant form g^{\mu\nu}

## arguments.csv

file with metric arguments - Schwarzschild radius, for example

## T

final `T` variable for calculations

## h

Runge-Kutta iteration step

## num_steps

number of Runge-Kutta steps between storing intermidiate results

## output dir

directory to save each geodesic full path

# Python wrapper

It emits geodesics for light rays

```
python3 geodesic_rt.py profile.yaml
```

`profile.yaml` describes calculation task:

```
---
rs: 1                 # schwarzschild radius
r0: 15                # observer position
fov: 6.282            # FOV ob observer (2 pi)
nrays: 640            # number of rays to emit
T: 100                # iteration limit
h: 5e-4               # iteration step
metric: schwarzschild # metric representation
```

# Image generator

Generate image

```
```