import math 

import pymeshfix
import numpy as np
import open3d as o3d
import pyvista as pv
import os

def euclidean_distance(p1, p2):
    return abs(math.sqrt(  ( (p1[0] - p2[0]) ** 2) + ( (p1[1] - p2[1]) ** 2) )  )


def restoration(src, filename, dir):

    gt_mesh = o3d.io.read_triangle_mesh(src)

    mesh = pv.read(src)

    verts = np.asarray(gt_mesh.vertices)

    surf = pv.PolyData(verts, mesh.faces)

    surf.save(f"{dir}/{filename.strip('.PLY')}-polydata.ply")

    # Read mesh from infile and output cleaned mesh to outfile
    pymeshfix.clean_from_file(f'{dir}/{filename.strip(".PLY")}-polydata.ply', f'{dir}/{filename.strip(".PLY")}-fixed.ply')

    os.remove(f"{dir}/{filename.strip('.PLY')}-polydata.ply")