import traceback
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


    def get_wrist_circumference_path(self):
        lines = trimesh.intersections.mesh_plane(self.mesh,plane_normal=[0,-1,0],plane_origin=[self.keypoints['wrist'][0],self.keypoints['wrist'][1],0])
        p = trimesh.load_path(lines)
        return p


    def get_plane_from_crossing_side_common_points(self):
        p0 = self.keypoints['side1'] + [0]
        p1 = self.keypoints['side2'] + [2]
        p2 = (p0 - p1)/ 2 # just set third point to some random point, 

        p2[1] += 20 # just change the third point to move it outside of the p0-p1 line!
        
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
        vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]

        u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

        # print(f"p0 {p0}")
        # print(f"p1 {p1}")
        # print(f"p2 {p2}")
        # print(f"u_cross_v {u_cross_v}")

        point  = np.array(p0)
        normal = np.array(u_cross_v)
        normal[2] = 0 # set zero on Z axis so as to make cutting plane perpendicular to hand


        return np.array(p0), normal

    def get_max_circumference_path(self):

        point, normal = self.get_plane_from_crossing_side_common_points()

        lines = trimesh.intersections.mesh_plane(self.mesh,plane_normal=normal,plane_origin=point)
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


    def show_wrist_circumference_path(self):
        pl = pv.Plotter()
        path = self.get_wrist_circumference_path()
        pl.add_mesh(path.vertices, color=True)
        pl.add_mesh(self.mesh, color=True)
        # p.show_bounds(grid='front', location='outer', 
        #                             all_edges=True)
        pl.camera_position = 'xy'
        pl.camera.roll += 10
        pl.title = self.image_2d_info['filename']
        pl.show()

    def show_circumference_path(self):
        point, normal = self.get_cross_points_plane()

        p = pv.Plotter()
        path = self.get_max_circumference_path()
        p.add_mesh(path.vertices, color=True)
        p.add_mesh(self.mesh, color=True)
        # p.add_mesh(point, color=True,render_points_as_spheres=True,
        #               point_size=100.0)

        # p.add_mesh(pv.Plane(center=point,direction=normal,i_size=300,j_size=300))
        p.show_bounds(grid='front', location='outer', 
                                    all_edges=True)
        p.title = self.image_2d_info['filename']
        p.show()

    
    def get_max_circumference(self):
        try:
            p = self.get_max_circumference_path()
            p2 = trimesh.path.Path3D(entities=[p.entities[0]],vertices=p.vertices)
            if len(p.entities)  == 1:
                return p2.length
            p3 = trimesh.path.Path3D(entities=[p.entities[1]],vertices=p.vertices)
            return max(p2.length, p3.length)
        except IndexError as e:
            print(f"Index error for {self.image_2d_info['filename']}")
            traceback.print_exc()
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

        if not self.mesh.is_watertight:
            raise Exception("OOOps your mesh is not water tight")
        
        features['filename'] = self.image_2d_info['filename']
        features['volume'] = self.get_sliced_volume()
        features['wrist_circumference'] = self.get_wrist_circumference_path().length
        features['max_circumference'] = self.get_max_circumference()
        features['scale'] = self.scale
        features['max_diameter'] = self.get_wrist_depth()
        features['wrist_ratio'] = self.get_wrist_ratio()
        
        return features

if __name__ == '__main__':
    import json
    import os
    import pandas as pd
    MESHES_PATH = "./Data/new/fixed"
    OUTPUT_DF_PATH = "./output/results.csv"

    output_df = pd.read_csv("./output/results.csv")
    output_df = output_df.sort_values(by=['index'])
    with open('output/results.json') as json_file:
        data = json.load(json_file)

        error_count = 0

        for idx, (mesh_ID, mesh_data) in enumerate(data.items()):
            mesh_path = os.path.join(MESHES_PATH, mesh_ID+'.R-fixed.ply')
            mesh_ID = int(mesh_ID)
            # print(f'mesh_data : {mesh_data}')

            if len(mesh_data.keys()) > 0 :
                features = MeshFeatures(mesh_path, mesh_data,scale = 4)
                # features.show_keypoints()
                # features.show_wrist_circumference_path()
                # features.show_circumference_path()
                try:
                    features_dict = features.get_all_features_dict()
                    print(f"features : {features_dict}")

                    # if idx > 5:
                    #     break
                    
                    output_df.iloc[mesh_ID-1,output_df.columns.get_loc('volume')] = features_dict['volume']
                    output_df.iloc[mesh_ID-1,output_df.columns.get_loc('max_circumference')] = features_dict['max_circumference']
                    output_df.iloc[mesh_ID-1,output_df.columns.get_loc('wrist_circumference')] = features_dict['wrist_circumference']
                    output_df.iloc[mesh_ID-1,output_df.columns.get_loc('max_diameter')] = features_dict['max_diameter']
                    output_df.iloc[mesh_ID-1,output_df.columns.get_loc('wrist_ratio')] = features_dict['wrist_ratio']
                except Exception:
                    error_count = error_count+1
                    print(f"Could not create data for mesh {mesh_ID}, error_count{error_count }")

    output_df.to_csv(OUTPUT_DF_PATH)