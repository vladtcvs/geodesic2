import os
import tempfile
import pandas as pd
import subprocess
import math

CURDIR = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(CURDIR, "geodesic2")

def run_calculation(rays, metric, args, length, h):
    num_steps = 1000

    # saving initial to file
    input_file = tempfile.NamedTemporaryFile(mode='wt',delete=False)
    input_fname = input_file.name

    inp = rays.to_csv(sep=',', index=False, line_terminator='\n')
    input_file.write(inp)
    input_file.close()

    # saving args to file
    args_file = tempfile.NamedTemporaryFile(mode='wt',delete=False)
    args_fname = args_file.name

    args_csv = args.to_csv(sep=',', index=False, line_terminator='\n')
    args_file.write(args_csv)
    args_file.close()

    # create file for output
    output_file = tempfile.NamedTemporaryFile(mode='wt', delete=False)
    output_fname = output_file.name
    output_file.close()

    run_args = [
        BINARY,
        input_fname,                            # inital rays position & direction
        output_fname,                           # result of calculation
        os.path.join(CURDIR, metric + ".cl"),   # file with metric description
        args_fname,
        "%lf" % length,                         # length of integration
        "%lf" % h,                              # step of integration
        str(num_steps)                          # extract data from OpenCL every each `num_steps`
        ]

    print("Command: %s" % (' '.join(run_args)))

    subprocess.run(run_args)

    result = pd.read_csv(output_fname, sep=',')

    os.remove(input_fname)
    os.remove(output_fname)
    os.remove(args_fname)

    return result

def ray(r0, alpha, rs):
    k = 1 - rs/r0
    d = (k / (1/k * math.cos(alpha)**2 + math.sin(alpha)**2))**0.5
    dr = -d * math.cos(alpha)
    dphi = d * math.sin(alpha) / r0
    return (-1, dr, 0, dphi)

def init_rays_perspective(rs, r0, fov, nrays):
    columns = []
    dtypes = []

    for i in range(dimensions):
        columns.append('pos%i' % i)
        dtypes.append('float64')

    for i in range(dimensions):
        columns.append('dir%i' % i)
        dtypes.append('float64')

    rays = pd.DataFrame(columns=columns)

    w = math.tan(fov/2)

    for i in range(nrays):
        s = 2*i/(nrays-1)-1
        tan = w*s
        angle = math.atan(tan)
        (dt, dr, dtheta, dphi) = ray(r0, angle, rs)
        rays.loc[i] = [0.0, r0, math.pi/2, 0.0, dt, dr, dtheta, dphi]

    return rays

def init_rays_equal(rs, r0, fov, nrays):
    columns = []
    dtypes = []

    for i in range(dimensions):
        columns.append('pos%i' % i)
        dtypes.append('float64')

    for i in range(dimensions):
        columns.append('dir%i' % i)
        dtypes.append('float64')

    rays = pd.DataFrame(columns=columns)
    angles = pd.DataFrame(columns=['init_angle'])

    for i in range(nrays):
        angle = fov/2 * i / (nrays - 1)
        (dt, dr, dtheta, dphi) = ray(r0, angle, rs)
        rays.loc[i] = [0.0, r0, math.pi/2, 0.0, dt, dr, dtheta, dphi]
        angles.loc[i] = [angle]

    return rays, angles

def get_angle(rs, pos, dir):
    r = pos[1]
    phi = pos[3]

    dt = dir[0]
    dr = dir[1]
    dphi = dir[3]

    dr /= abs(dt)
    dphi /= abs(dt)

    alpha = math.atan2(r*dphi, -dr)
    gamma = math.pi - (phi + (math.pi - alpha))

    while gamma < -math.pi:
        gamma += math.pi*2
    while gamma > math.pi:
        gamma -= math.pi*2
    return gamma

def calculate(rs, rays, T, h):
    metric = 'schwarzschild'

    args = pd.DataFrame(columns=['arg'])
    args.loc[0] = [rs]

    final = run_calculation(rays, metric, args, T, h)
    return final

dimensions = 4
rs = 1
r0 = 15
fov = math.pi
pixels = 500
T = 300
h = 5e-5

init_rays = init_rays_equal

rays, angles = init_rays(rs, r0, fov, pixels)
final = calculate(rs, rays, T, h)

angles['final_angle'] = pd.Series(0.0, index=angles.index)
angles['collided'] = final['collided']

for pix in range(pixels):
    if not final.loc[pix]['collided']:
        pos = [final.loc[pix]['pos%i' % i] for i in range(dimensions)]
        dir = [final.loc[pix]['dir%i' % i] for i in range(dimensions)]

        angles.at[pix, 'final_angle'] = get_angle(rs, pos, dir)

rays.to_csv("input.csv", sep=',', index=False, line_terminator='\n')
final.to_csv("output.csv", sep=',', index=False, line_terminator='\n')
angles.to_csv("angles.csv", sep=',', index=False, line_terminator='\n')
