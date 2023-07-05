import ast
import trimesh
import numpy as np
import pandas as pd
import pyvista as pv


def get_transformed_keypoints_from_2d_on_mesh(mesh, image_info, scale=None):
    print(f"image_info {image_info}")
    a_index = np.argmax(mesh.vertices, axis=0)[0] # thumb
    b_index = np.argmin(mesh.vertices, axis=0)[1] # top
    
    mesh_A_point = mesh.vertices[a_index][:2]
    mesh_B_point = mesh.vertices[b_index][:2]
    
    image_B_pixel = np.array(image_info["bottom"])[::-1]
    image_B_pixel[1] = max(image_info["shape"]) - image_B_pixel[1]

    keypoint_names = ["wrist","thumb","little","side1","side2","side4","side5","side6","end_middle","index_finger","end_index","ring","end_ring"]
    keypoints = {}


    keypoints["top"] = image_B_pixel

    thumb = None
    for name in keypoint_names:

        keypoint = image_info[name]
        
        try:
            keypoint[1] = max(image_info["shape"]) - keypoint[1]

            if name == "thumb":
                thumb = keypoint
            keypoints[name] = keypoint
        except TypeError:
            print(f'ERROR: keypoint {name} not found!')

    image_A_pixel = thumb
    
    
    points_distance_image =  np.linalg.norm(image_A_pixel-image_B_pixel)
    points_distance_mesh =  np.linalg.norm(mesh_A_point-mesh_B_point)
    
    image_to_mesh_scale = points_distance_image / points_distance_mesh

    if scale:
        image_to_mesh_scale = scale
    
    mesh_points_scaled = mesh.vertices.copy() * image_to_mesh_scale
    
    scaled_mesh_A_point = mesh_points_scaled[a_index][:2]
    
    scaled_mesh_B_point = mesh_points_scaled[b_index][:2]
    
    translation_offset =  scaled_mesh_B_point - image_B_pixel
    
    for keypoint_name, keypoint in keypoints.items():
        keypoint_location_on_mesh = (translation_offset + keypoint) / image_to_mesh_scale
        keypoint_location_on_mesh = np.array(list(keypoint_location_on_mesh) + [50])
        keypoints[keypoint_name] = keypoint_location_on_mesh

    
    return keypoints, translation_offset, image_to_mesh_scale



