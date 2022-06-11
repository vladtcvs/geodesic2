import pandas as pd
import math
import numpy as np
import imageio
import sys

import multiprocessing as mp

import yaml

def normalize_block(block):
    nrise = 0
    nfall = 0

    for i in range(10, len(block)):
        if block[i] > block[i-10]:
            nrise += 1
        else:
            nfall += 1

    if nrise > nfall:
        rising = True
    else:
        rising = False

    for i in range(1, len(block)):
        if rising:
            while block[i] < block[i-1]:
                block[i] += 2*math.pi
        else:
            while block[i] > block[i-1]:
                block[i] -= 2*math.pi
    return block

def prepare_table(angles_table):
    prev_angle = None
    begin_angle = None
    blocks = []
    block = []
    world_id = None
    for i in range(angles_table.shape[0]):
        collided = angles_table.loc[i]['collided']
        if collided:
            if len(block) > 1:
                nblock = normalize_block(block)
                blocks.append((begin_angle, prev_angle, world_id, nblock))
            block = []
        else:
            if len(block) == 0:
                begin_angle = angles_table.loc[i]['init_angle']
            block.append(angles_table.loc[i]['final_angle'])
            prev_angle = angles_table.loc[i]['init_angle']
        world_id = angles_table.loc[i]['world']

    if len(block) > 1:
        nblock = normalize_block(block)
        blocks.append((begin_angle, prev_angle, world_id, nblock))

    return blocks

def approximate_angle(angles_table_blocks, angle):
    for block in angles_table_blocks:
        #print(block)
        begin = block[0]
        end = block[1]
        world_id = block[2]
        angles = block[3]

        if angle < begin:
            continue
        if angle > end:
            continue

        pos = (angle - begin) / (end - begin) * (len(angles) - 1)

        ind1 = int(math.floor(pos))
        ind2 = int(math.ceil(pos))

        #print(angle, begin, end, pos, len(angles))

        if ind1 == ind2:
            return angles[ind1], False, world_id
        else:
            da = ind2 - ind1
            da1 = pos - ind1
            da2 = ind2 - pos
            return (angles[ind1] * da2 + angles[ind2] * da1)/da, False, world_id
    return 0, True, 0

def normal(a, b):
    s = [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    ns = (s[0]**2 + s[1]**2 + s[2]**2)**0.5
    return [s[0]/ns, s[1]/ns, s[2]/ns]

def rotate(vec, axis, angle):
    p = normal(axis, vec)
    res = [0, 0, 0]
    for i in range(3):
        res[i] = vec[i]*math.cos(angle) + p[i]*math.sin(angle)
    return res

def direction(vec):
    z = vec[2]
    x = vec[0]
    y = vec[1]
    theta = math.acos(z)
    phi = math.atan2(y, x)
    return theta, phi

def getpixel(theta, phi, image):
    while phi < 0:
        phi += math.pi * 2
    while phi > 2 * math.pi:
        phi -= math.pi * 2

    H = image.shape[0]
    W = image.shape[1]
    
    x = phi / (2*math.pi) * W
    
    xm = math.floor(x)
    xp = math.ceil(x)

    xmc = 1-(x-xm)
    xpc = 1-(xp-x)

    while xp < 0:
        xp += W

    while xm < 0:
        xm += W
    
    while xm >= W:
        xm -= W

    while xp >= W:
        xp -= W

    c = xmc + xpc
    xmc /= c
    xpc /= c

    y = theta / math.pi * H

    ym = math.floor(y)
    yp = math.ceil(y)

    ymc = 1-(y-ym)
    ypc = 1-(yp-y)

    if ym < 0:
        ym = 0
    
    if yp < 0:
        yp = 0

    if ym >= H:
        ym = H - 1

    if yp >= H:
        yp = H - 1

    c = ymc + ypc
    ymc /= c
    ypc /= c

    imm = image[ym][xm]
    imp = image[ym][xp]
    ipm = image[yp][xm]
    ipp = image[yp][xp]

    return imm * ymc*xmc + imp * ymc*xpc + ipm * ypc*xmc + ipp * ypc*xpc

def calculate_pixel(img, Y, X, H, W, base_phi, universes):
    TH = Y / H * math.pi
    PHI = X / W * 2 * math.pi + base_phi

    vec = [math.sin(TH)*math.cos(PHI), math.sin(TH)*math.sin(PHI), math.cos(TH)]

    angle_initial = math.acos(e[0]*vec[0] + e[1]*vec[1] + e[2]*vec[2])
    angle_final, collided, world = approximate_angle(angle_blocks, angle_initial)

    if collided:
        img[Y,X,1] = 0
    else:
        da = angle_final - angle_initial

        n = normal(vec, e)
        vv = rotate(vec, n, -da)

        theta, phi = direction(vv)

        if world in universes:
            img[Y, X] = getpixel(theta, phi + universes[world]["phi"], universes[world]["skymap"])
        else:
            img[Y, X] = 0

with open(sys.argv[1]) as f:
    profile  = yaml.safe_load(f)

H = int(profile["imager"]["H"])
W = H*2

img = np.zeros((H, W, 3))

angles  = pd.read_csv(profile["scene"]["files"]["angles"], sep=',')

angle_blocks = prepare_table(angles)

universes_desc  = profile["imager"]["universes"]
base_phi   = float(profile["imager"]["viewer_orientation"])*math.pi/180

universes = {}
for id in universes_desc:
    fname = universes_desc[id]["skymap"]
    orientation = float(universes_desc[id]["orientation"])*math.pi/180
    skymap = np.asarray(imageio.imread(fname))
    skymap = skymap / np.amax(skymap)
    universes[id] = {"phi" : orientation, "skymap" : skymap}

e = [1, 0, 0]

args = []
for Y in range(H):
    print("%i / %i" % (Y, H))
    for X in range(W):
        calculate_pixel(img, Y, X, H, W, base_phi, universes)

imageio.imwrite(profile["imager"]["output"], img)

