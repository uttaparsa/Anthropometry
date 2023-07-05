import traceback
import json
import re
import os
from mesh_3d_features import MeshFeatures, MESHES_PATH
from utils import euclidean_distance
import cv2
import numpy as np
import pandas as pd

POINT_NAMES_BY_NUMBER = {
    "0" : "top",
    "1" : "wrist",
    "2" : "thumb",
    "3" : "little",
    "4" : "side1",
    "5" : "side2",
    "6" : "end_middle",
    "7" : "side3",
    "8" : "side4",
    "9" : "side5",
    "10" : "side6",
    "11" : "index_finger",
    "12" : "end_index",
    "13" : "ring",
    "14" : "end_ring",
}

def get_top_and_bottom_and_shape(mesh_id):
    img = cv2.imread('./raw_images_new/images/'+mesh_id+".R.png",0)
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    black_pixels = np.argwhere(thresh1 == 0)
    max_height_row = black_pixels[:, 0].max()

    # Find the column with the highest black pixel in that row
    max_height_row_pixels = black_pixels[black_pixels[:, 0] == max_height_row]
    max_height_col = max_height_row_pixels[:, 1].max()

    # The coordinates of the black pixel with the highest height
    max_height_pixel = [max_height_row, max_height_col]

    min_height_row = black_pixels[:, 0].min()

    # Find the column with the lowest black pixel in that row
    min_height_row_pixels = black_pixels[black_pixels[:, 0] == min_height_row]
    min_height_col = min_height_row_pixels[:, 1].max()

    # The coordinates of the black pixel with the loest height
    min_height_pixel = [min_height_row, min_height_col]

    return max_height_pixel, min_height_pixel, thresh1.shape


def convert_points_to_named_format(points,mesh_id):
    result_points = {'filename':mesh_id+'.R.png'}
    result_points["bottom"], result_points["top"], result_points["shape"]  = get_top_and_bottom_and_shape(mesh_id)
    for idx,point in enumerate(points):
        
        result_points[POINT_NAMES_BY_NUMBER[str(point["id"])]] = [point["x"],point["y"]]
    return result_points

def convert_points_to_array_format(points ):
    result = []
    for p in   sorted(points, key=lambda d: d['id']) :
        result.append([p['x'],p['y']])
    return result



with open('./Data/points_dict.json') as json_file:
    data : dict = json.load(json_file)

    df_list = []
    error_count = 0
    pDic = dict()
    for file_name, points in data.items():
        file_number = re.findall(r'\d+', file_name)[0]
        print(f"file_number is {file_number}")
        mesh_path =  os.path.join(MESHES_PATH, file_number+'.R-fixed.ply')
        
        points_in_the_other_format = convert_points_to_named_format(points,file_number)
        if len(points_in_the_other_format.keys()) < 8:
            continue
        if file_number != '83':
            continue
        try:
            mf = MeshFeatures(mesh_path,points_in_the_other_format,scale=4)
        except:
            print(f"ERRRROR for {file_number}")
            continue

        points_array_format = convert_points_to_array_format(points)
        # print(f"points_array_format {points_array_format}")
        # dst = euclidean_distance(points_array_format[0], points_array_format[6], 0.25)
        # print("distance is "+str(dst))
        mf.show_keypoints()
        mf.show_wrist_circumference_path()
        # break
        pDic = dict()
        pDic["hand_length"]   = euclidean_distance(points_array_format[0], points_array_format[1], 0.25)
        pDic["palm_length"]   = euclidean_distance(points_array_format[1], points_array_format[6], 0.25)
        pDic["finger_length"] = euclidean_distance(points_array_format[0], points_array_format[6], 0.25)
        pDic["index_length"] = euclidean_distance(points_in_the_other_format["index_finger"], points_in_the_other_format["end_index"], 0.25) if ("index_finger" in points_in_the_other_format) and ("end_index" in points_in_the_other_format) else None
        pDic["ring_length"] = euclidean_distance(points_in_the_other_format["ring"], points_in_the_other_format["end_ring"], 0.25) if ("ring" in points_in_the_other_format) and ("end_ring" in points_in_the_other_format) else None
        pDic["hand_width"]    = euclidean_distance(points_array_format[4], points_array_format[5], 0.25)
        pDic["index1"] = pDic["hand_width"] * (100 / pDic["hand_length"])
        pDic["max_breadth"] = euclidean_distance(points_array_format[7], points_array_format[8], 0.25)
        pDic["hand_span"] = euclidean_distance(points_array_format[2], points_array_format[3], 0.25)



        pDic['index'] = file_number
        pDic["volume"] = None
        pDic["index2"] = None
        pDic["max_circumference"] = None
        pDic["max_diameter"] = None
        pDic["wrist_ratio"] = None
        pDic["hLength_bHeight"] = None
        pDic["hVolume_BMI"] = None

        if len(points_in_the_other_format.keys()) > 0 :

            try:
                features_dict = mf.get_all_features_dict()
                print(f"features : {features_dict}")

                # df.loc[df['column_name'] == some_value]
                # print(f"we are mofifying {output_df.iloc[mesh_ID-1]}")
                
                pDic['volume']  = features_dict['volume']
                pDic[ 'max_circumference'] = features_dict['max_circumference']
                pDic[  'wrist_circumference'] = features_dict['wrist_circumference']
                pDic[ 'max_diameter'] = features_dict['max_diameter']
                pDic[  'wrist_ratio'] = features_dict['wrist_ratio']

                # output_df.iloc[mesh_ID-1,output_df.columns.get_loc('volume')] = features_dict['volume']
                # output_df.iloc[mesh_ID-1,output_df.columns.get_loc('max_circumference')] = features_dict['max_circumference']
                # output_df.iloc[mesh_ID-1,output_df.columns.get_loc('wrist_circumference')] = features_dict['wrist_circumference']
                # output_df.iloc[mesh_ID-1,output_df.columns.get_loc('max_diameter')] = features_dict['max_diameter']
                # output_df.iloc[mesh_ID-1,output_df.columns.get_loc('wrist_ratio')] = features_dict['wrist_ratio']
            except Exception:
                error_count = error_count + 1
                print(f"Could not create data for mesh {file_number}, error_count is {error_count }")
                print(traceback.format_exc())
            

            df_list.append(pDic);
    df_result = pd.DataFrame.from_dict(df_list)
    df_result.to_csv("results.csv")


        
