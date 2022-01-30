import os
import tempfile
import pandas as pd
import subprocess
import math

import lemaitre
import kruskal
import schwarzschild

ray = {
    "schwarzschild" : schwarzschild.ray,
    "lemaitre" : lemaitre.ray,
    "kruskal" : kruskal.ray,
}

transform_to = {
    "schwarzschild" : schwarzschild.transform_to,
    "lemaitre" : lemaitre.transform_to,
    "kruskal" : kruskal.transform_to,
}

transform_from = {
    "schwarzschild" : schwarzschild.transform_from,
    "lemaitre" : lemaitre.transform_from,
    "kruskal" : kruskal.transform_from,
}

collision = {
    "schwarzschild" : schwarzschild.collision,
    "lemaitre" : lemaitre.collision,
    "kruskal" : kruskal.collision,
}

def build_ray(metric, r0, alpha, rs):
    return ray[metric](r0, alpha, rs)

def transform_from_target(metric, pos, dir, rs):
    return transform_from[metric](pos, dir, rs)

def transform_to_target(metric, pos, dir, rs):
    return transform_to[metric](pos, dir, rs)

def check_collision(metric, pos, rs):
    return collision[metric](pos, rs)

CURDIR = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(CURDIR, "geodesic2")

def run_calculation(rays, metric, args, length, h, save_rays_dir):
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
        str(num_steps),                         # extract data from OpenCL every each `num_steps`
        ]
    if save_rays_dir is not None:
        run_args.append(save_rays_dir)

    print("Command: %s" % (' '.join(run_args)))

    subprocess.run(run_args)

    print("input:  %s" % input_fname)
    print("output: %s" % output_fname)

    types = {}
    
    types['collided'] = "bool"
    for i in range(4):
        types['pos%i' % i] = 'float'
        types['dir%i' % i] = 'float'

    result = pd.read_csv(output_fname, sep=',', dtype=types)

    os.remove(input_fname)
    os.remove(output_fname)
    os.remove(args_fname)
    return result

def init_rays(rs, r0, fov, nrays, metric):
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

    nr = 0
    for i in range(nrays):
        angle = fov/2 * i / (nrays - 1)

        valid, pos, dir = build_ray(metric, r0, angle, rs)

        maxd = max([abs(item) for item in dir])
        if maxd > 10:
            k = maxd / 10
            dir = [item/k for item in dir]

        if valid:
            rays.loc[nr] = [pos[0], pos[1], pos[2], pos[3], dir[0], dir[1], dir[2], dir[3]]
            angles.loc[nr] = [angle]
            nr += 1

    return rays, angles

def get_output_angle(metric, pos, dir, rs):
    valid, pos, dir, attrs = transform_from_target(metric, pos, dir, rs)
    if not valid:
        return False, 0, None

    r = pos[1]
    phi = pos[3]

    dt = dir[0]
    dr = dir[1]
    dphi = dir[3]

    dr /= abs(dt)
    dphi /= abs(dt)

    alpha = math.atan2((r-rs)*dphi, -dr)
    gamma = math.pi - (phi + (math.pi - alpha))

    while gamma < -math.pi:
        gamma += math.pi*2
    while gamma > math.pi:
        gamma -= math.pi*2

    if metric == "kruskal":
        area = attrs["area"]
        if area == 1:
            world = 1
        elif area == 3:
            world = 2
        else:
            world = None
    else:
        world = 1
    return True, gamma, world

def calculate_rays(rs, rays, T, h, metric, save_rays_dir):
    args = pd.DataFrame(columns=['arg'])
    args.loc[0] = [rs]
    final = run_calculation(rays, "metrics/" + metric, args, T, h, save_rays_dir)
    return final

dimensions = 4
rs = 1
r0 = 15
fov = math.pi*2
pixels = 3000
T = 100
h = 5e-4
metric = 'schwarzschild'
#metric = 'lemaitre'
#metric = 'kruskal'

#save_rays_dir = 'rays'
save_rays_dir = None

rays, angles = init_rays(rs, r0, fov, pixels, metric)
final = calculate_rays(rs, rays, T, h, metric, save_rays_dir)

angles['final_angle'] = pd.Series(0.0, index=angles.index)
angles['collided'] = pd.Series(False, index=angles.index)

for pix in range(pixels):
    pos = [final.loc[pix]['pos%i' % i] for i in range(dimensions)]
    dir = [final.loc[pix]['dir%i' % i] for i in range(dimensions)]

    is_collided = check_collision(metric, pos, rs)

    if is_collided:
        angles.at[pix, 'collided'] = True
        angles.at[pix, 'world'] = -1
    else:
        valid, gamma, world = get_output_angle(metric, pos, dir, rs)

        if valid:
            angles.at[pix, 'final_angle'] = gamma
            angles.at[pix, 'world'] = world
        else:
            angles.at[pix, 'collided'] = True
            angles.at[pix, 'world'] = -1

rays.to_csv("input.csv", sep=',', index=False, line_terminator='\n')
final.to_csv("output.csv", sep=',', index=False, line_terminator='\n')
angles.to_csv("angles.csv", sep=',', index=False, line_terminator='\n')
