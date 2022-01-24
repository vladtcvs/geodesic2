import pandas as pd
import math
import numpy as np
import imageio
import sys

def normalize_block(block):
    for i in range(1, len(block)):
        while block[i] < block[i-1]:
            block[i] += 2*math.pi
    return block

def prepare_table(angles_table):
    prev_angle = None
    begin_angle = None
    blocks = []
    block = []
    for i in range(angles_table.shape[0]):
        collided = angles_table.loc[i]['collided']
        if collided:
            if len(block) > 1:
                blocks.append((begin_angle, prev_angle, normalize_block(block)))
            block = []
        else:
            if len(block) == 0:
                begin_angle = angles_table.loc[i]['init_angle']
            block.append(angles_table.loc[i]['final_angle'])
            prev_angle = angles_table.loc[i]['init_angle']
    if len(block) > 1:
        blocks.append((begin_angle, prev_angle, normalize_block(block)))
        
    return blocks

def approximate_angle(angles_table_blocks, angle):
    for block in angles_table_blocks:
        begin = block[0]
        end = block[1]
        angles = block[2]

        if angle < begin:
            continue
        if angle > end:
            continue

        pos = (angle - begin) / (end - begin) * (len(angles) - 1)

        ind1 = int(math.floor(pos))
        ind2 = int(math.ceil(pos))

        #print(angle, begin, end, pos, len(angles))

        if ind1 == ind2:
            return angles[ind1], False
        else:
            da = ind2 - ind1
            da1 = pos - ind1
            da2 = ind2 - pos
            return (angles[ind1] * da2 + angles[ind2] * da1)/da, False
    return 0, True

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

# resulting image size
fov = 2*math.pi/3
W = 15000
H = 7500

img = np.zeros((H, W, 3))

w = 2 * math.tan(fov/2)
h = 2 * math.tan(fov/2)

rays = pd.read_csv('input.csv', sep=',')
final = pd.read_csv('output.csv', sep=',')
angles  = pd.read_csv("angles.csv", sep=',')

angle_blocks = prepare_table(angles)

#print(approximate_angle(angle_blocks, 0.3))

#sys.exit()

sky_fname = "sky_map.png"
base_phi = math.pi
view_phi = 120 * math.pi / 180

sky = np.asarray(imageio.imread(sky_fname))
sky = sky / np.amax(sky)


e = [1, 0, 0]

for Y in range(H):
    TH = Y / H * math.pi
    print("%i / %i" % (Y, H))
    for X in range(W):
        #print("%i / %i" % (X, W))
        PHI = X / W * 2 * math.pi + base_phi

        #img[Y, X] = getpixel(TH, PHI, sky)
        ##continue

        vec = [math.sin(TH)*math.cos(PHI), math.sin(TH)*math.sin(PHI), math.cos(TH)]

        angle_initial = math.acos(e[0]*vec[0] + e[1]*vec[1] + e[2]*vec[2])
        angle_final, collided = approximate_angle(angle_blocks, angle_initial)

        #if angle_initial > fov:
        #    valid = False
        #else:
        #    angle_final = angle_initial / 2
        #    collided = False
        #    valid = True

        if collided:
            img[Y,X,1] = 0
        else:
            da = angle_final - angle_initial

            n = normal(vec, e)
            vv = rotate(vec, n, -da)

            theta, phi = direction(vv)

            img[Y, X] = getpixel(theta, phi + view_phi, sky)

imageio.imwrite("black_hole.png", img)
