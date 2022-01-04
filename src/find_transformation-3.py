#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ast
import trimesh
import numpy as np
import pandas as pd
import pyvista as pv


# In[2]:


df = pd.read_csv('/home/parsa/new_volume/DARSEPORJE/Anthropometry/src/output/keypoints.csv') 
image_info = df.iloc[0]
print(image_info)

mesh = trimesh.load("/home/parsa/new_volume/DARSEPORJE/Anthropometry/src/Data/fixed/"+image_info['filename']+".ply", force='mesh')


# In[3]:


max_y_index = np.argmax(mesh.vertices, axis=0)[1]
max_y_index


# In[4]:


min_y_index = np.argmin(mesh.vertices, axis=0)[1]
min_y_index


# In[5]:


mesh_top_point = mesh.vertices[max_y_index][:2]
mesh_top_point


# In[6]:


mesh_bottom_point = mesh.vertices[min_y_index][:2]
mesh_bottom_point


# In[7]:


image_bottom_pixel =  np.array(ast.literal_eval(image_info["bottom"]))[::-1]
image_top_pixel = np.array(ast.literal_eval(image_info["top"]))[::-1]
image_bottom_pixel[1] = 944 - image_bottom_pixel[1]
image_top_pixel[1] = 944 - image_top_pixel[1]


keypoint_names = ["wrist","thumb","little","side1","side2","side4","side5","side6","end_middle"]
keypoints = []

keypoints.append({"name":"top", "pixel_location": image_top_pixel})
keypoints.append({"name":"bottom", "pixel_location": image_bottom_pixel})


for name in keypoint_names:
    keypoint = {}
    keypoint["name"] = name
    keypoint["pixel_location"] = np.array(ast.literal_eval(image_info[name]))
    keypoint["pixel_location"][1] = 944 - keypoint["pixel_location"][1]
    keypoints.append(keypoint)


# In[8]:


points_distance_image =  np.linalg.norm(image_top_pixel-image_bottom_pixel)
points_distance_image


# In[9]:


points_distance_mesh =  np.linalg.norm(mesh_top_point-mesh_bottom_point)
points_distance_mesh


# In[10]:


image_to_mesh_scale = points_distance_image / points_distance_mesh
image_to_mesh_scale


# In[11]:


mesh_points_scaled = mesh.vertices.copy() * image_to_mesh_scale
mesh_points_scaled


# In[12]:


scaled_mesh_top_point = mesh_points_scaled[max_y_index][:2]
scaled_mesh_top_point


# In[13]:


scaled_mesh_bottom_point = mesh_points_scaled[min_y_index][:2]
scaled_mesh_bottom_point


# In[14]:


translation_offset =  scaled_mesh_bottom_point - image_bottom_pixel
translation_offset


# In[15]:


keypoints


# In[16]:


pv.set_jupyter_backend(None)
p = pv.Plotter()

p.add_mesh(mesh, color=True)

for keypoint in keypoints:
    keypoint_location_on_mesh = (translation_offset + keypoint['pixel_location']) / image_to_mesh_scale
    
    keypoint_location_on_mesh = np.array(list(keypoint_location_on_mesh) + [50])
    temp_point = keypoint_location_on_mesh.copy()
    temp_point[2] -= 100
    print(temp_point)
    print(keypoint_location_on_mesh)
    line = pv.Line(keypoint_location_on_mesh,temp_point)
    p.add_mesh(line, color="b",line_width=1)
    # p.add_mesh(keypoint_location_on_mesh, color="red", line_width=10)


p.show()


# In[ ]:




