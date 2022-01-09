import trimesh
import numpy as np
import pandas as pd
import pyvista as pv

import find_transformation


class MeshFeatures():
    def __init__(self, mesh_path,image_2d_info,scale=None) -> None:
        self.image_2d_info = image_2d_info
        self.mesh = trimesh.load(mesh_path, force='mesh')
        self.keypoints, self.translation_offset, self.scale = find_transformation.get_transformed_keypoints_from_2d_on_mesh(self.mesh, image_2d_info,scale=scale)


    def get_wrist_circumference(self):
        lines = trimesh.intersections.mesh_plane(self.mesh,plane_normal=[0,-1,0],plane_origin=[self.keypoints['wrist'][0],self.keypoints['wrist'][1],0])
        p = trimesh.load_path(lines)
        return p.length

    def get_max_circumference_path(self):
        lines = trimesh.intersections.mesh_plane(self.mesh,plane_normal=[0,-1,0],plane_origin=[self.keypoints['side1'][0],self.keypoints['side1'][1],0])
        p = trimesh.load_path(lines)
        return p

    def show_keypoints(self):
        p = pv.Plotter()

        p.add_mesh(self.mesh, color=True)

        for keypoint_name, keypoint in self.keypoints.items():

            temp_point = keypoint.copy()
            temp_point[2] -= 100
            line = pv.Line(keypoint,temp_point)

            p.add_mesh(line, color="b",line_width=1)
            p.show_bounds(grid='front', location='outer', 
                                    all_edges=True)


        p.show()



    def show_circumference_path(self):
        p = pv.Plotter()
        path = self.get_max_circumference_path()
        p.add_mesh(path.vertices, color=True)
        p.add_mesh(self.mesh, color=True)
        p.show_bounds(grid='front', location='outer', 
                                    all_edges=True)
        p.show()

    
    def get_max_circumference(self):
        try:
            p = self.get_max_circumference_path()
            p2 = trimesh.path.Path3D(entities=[p.entities[0]],vertices=p.vertices)
            p3 = trimesh.path.Path3D(entities=[p.entities[1]],vertices=p.vertices)
            return max(p2.length, p3.length)
        except IndexError:
            return 0

    def get_wrist_width(self):
        return np.linalg.norm(
            np.array(self.keypoints['side5'])-np.array(self.keypoints['side6'])
            )

    def get_wrist_depth(self):
        return self.mesh.bounds[1][2] - self.mesh.bounds[0][2]

    def get_wrist_ratio(self):
        return self.get_wrist_depth() / self.get_wrist_width()

    def get_sliced_volume(self):
        sliced = trimesh.intersections.slice_mesh_plane(self.mesh,plane_normal=[0,-1,0],plane_origin=[self.keypoints['wrist'][0],self.keypoints['wrist'][1], 0],cap=True)
        return sliced.volume

    def get_all_features_dict(self):
        features = {}
        
        features['filename'] = self.image_2d_info['filename']
        features['volume'] = self.get_sliced_volume()
        features['wrist_circumference'] = self.get_wrist_circumference()
        features['max_circumference'] = self.get_max_circumference()
        features['scale'] = self.scale
        features['max_diameter'] = self.get_wrist_depth()
        features['wrist_ratio'] = self.get_wrist_ratio()
        
        return features

if __name__ == '__main__':
    import json
    import os
    import pandas as pd
    MESHES_PATH = "./Data/fixed"
    OUTPUT_DF_PATH = "./output/results.csv"

    output_df = pd.read_csv("./output/results.csv")
    output_df = output_df.sort_values(by=['index'])
    with open('output/results.json') as json_file:
        data = json.load(json_file)

        for mesh_name, mesh_data in data.items():
            mesh_path = os.path.join(MESHES_PATH, mesh_name+'-fixed.ply')
            mesh_name = int(mesh_name)
            print(f'mesh_data : {mesh_data}')
            if len(mesh_data.keys()) > 0:
                features = MeshFeatures(mesh_path, mesh_data,scale = 4)
                # features.show_keypoints()
                features_dict = features.get_all_features_dict()
                print(f"features : {features_dict}")
                output_df.iloc[mesh_name-1,output_df.columns.get_loc('volume')] = features_dict['volume']
                output_df.iloc[mesh_name-1,output_df.columns.get_loc('max_circumference')] = features_dict['max_circumference']
                output_df.iloc[mesh_name-1,output_df.columns.get_loc('wrist_circumference')] = features_dict['wrist_circumference']
                output_df.iloc[mesh_name-1,output_df.columns.get_loc('max_diameter')] = features_dict['max_diameter']
                output_df.iloc[mesh_name-1,output_df.columns.get_loc('wrist_ratio')] = features_dict['wrist_ratio']

    output_df.to_csv(OUTPUT_DF_PATH)