import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
import trimesh
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

from projection import project
from resortation import plyFix
from silhouette import silhouetter
from skeletone import Skeletone
from utils import (blockPrint, directory_handler, enablePrint,
                   euclidean_distance)

# Define the arguments being used to run the code
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--ply", 
	help="This is used for .ply files as input files.",
    action="store_true",
    default=False
)

ap.add_argument("-r", "--rotate", 
	help="rotate images -90 degree",
    action="store_true",
    default=False
)

ap.add_argument("-f", "--fix", 
	help="fix the ply input images",
    action="store_true",
    default=False
)

ap.add_argument("-s", "--scale", type=float, required=True,
	help="Scale of the measurements")

ap.add_argument("-i", "--input", type=str, required=True,
	help="path to the directory of the input images or ply files")

ap.add_argument("-o", "--output", type=str, required=True,
	help="path to the output directory")

args = vars(ap.parse_args())

# Define directories engaged in the process
input_dir = args['input']
output_dir = args['output']
cache_dir = "./cache"
cache_fix = "./cache/fix"
cache_image = "./cache/images"

# Checks the directory tree and creates necessary ones if there is a lack of them
directory_handler(args, input_path=input_dir, output_path=output_dir, cache_path=cache_dir)

# define the columns of the pandas Dataframe
df = pd.DataFrame(columns = [ 
                        "index",
                        "volume",
                        "hand_length",
                        "palm_length",
                        "finger_length",
                        "hand_width",
                        "little_length"
                        "ring_length"
                        "index1",
                        "index2",
                        "max_breadth",
                        "max_circumference",
                        "max_diameter",
                        "wrist_circumference",
                        "wrist_ratio",
                        "hLength_bHeight",
                        "hVolume_BMI"
                        "hand_span",
                        "bug",
                        "check"
                    ])

img_dir = input_dir

if args['ply']:
    img_dir = cache_image

    if args['fix']:
        plyFiles = next(os.walk(input_dir), (None, None, []))[2]

        # walks through the given PLY files and generates fixed PLY ones in the ./cache/fix directory
        print("Generating fixed ply files ...")
        # blockPrint()
        for pFile in plyFiles:
            mesh_path = f'{input_dir}/{pFile}'
            plyFix(mesh_path, filename=pFile, dir=cache_fix)
        
        # enablePrint()

        # Convert(project) the ply files into 2D images
        fixed_filenames = next(os.walk(cache_fix), (None, None, []))[2]
        # walk through the fixed PLY files and generates the 2D image files in ./cache/image directory
        for fixed_file in fixed_filenames:
            project(img_path=f"{cache_fix}/{fixed_file}", output_path=cache_image)
    else:
        # Convert(project) the ply files into 2D images
        fixed_filenames = next(os.walk(input_dir), (None, None, []))[2]
        # walk through the fixed PLY files and generates the 2D image files in ./cache/image directory
        for fixed_file in fixed_filenames:
            project(img_path=f"{input_dir}/{fixed_file}", output_path=cache_image)

# walk through the hand image files and find their points
filenames = next(os.walk(img_dir), (None, None, []))[2]  # [] if no file
points_dict = dict()
point20_bug_count = 0
empty_points_bug_count = 0
for file in filenames:
    print("file: ", file)
    pDic = dict()

    img = cv2.imread(f"{img_dir}/{file}")

    if not args["ply"]:
        if args['rotate']:
            sil_img = cv2.rotate(sil_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    sil_img = silhouetter(img)

    sk = Skeletone(image=sil_img, name=file.split(".")[0])
    bug = False
    check = False

    points = sk.keypoints()

    b_dic = b_list = None

    

    if points != None:
        print(f"points[20] : {points[20]}")
        if  points[0]  != None and \
            points[2]  != None and \
            points[4]  != None and \
            points[5]  != None and \
            points[9]  != None and \
            points[12] != None and \
            points[13] != None :

            b_dic, b_list = sk.border_points(points=points)

            
        elif points[20]:
            check = True
            print(f"points : {points}")
            print(f"points len : {len(points)}")
            bug = True
            point20_bug_count += 1
            # print("bug because of point20")
    else:
        bug = True
        # print("bug because of all")
        empty_points_bug_count += 1

    print(f"empty_points_bug_count {empty_points_bug_count}")
    print(f"point20_bug_count {point20_bug_count}")


    # add border points of the image to the point_dict. This dictionary will be converted to JSON file later.
    if b_list == None:
        points_dict[file.split('.')[0].split('-')[0]] = {}
    else:
        points_dict[file.split('.')[0].split('-')[0]] = {**b_dic, **(sk.get_top_bottom()) , **{'shape': img.shape,'filename':file}}
    
    print(f"b_dic : {b_dic}")

    if not bug:
        b_img = sk.draw_border_points()
        
        if b_img is not None:
            cv2.imwrite(f"{output_dir}/{file.strip('.ply').strip('.jpg').strip('.png').strip('.PLY')}.jpg", b_img)

        if b_list != None:
            pDic["hand_length"]   = euclidean_distance(b_list[0], b_list[1], args["scale"])
            pDic["palm_length"]   = euclidean_distance(b_list[1], b_list[6], args["scale"])
            pDic["finger_length"] = euclidean_distance(b_list[0], b_list[6], args["scale"])
            pDic["index_length"] = euclidean_distance(b_dic["index_finger"], b_dic["end_index"], args["scale"]) if ("index_finger" in b_dic) and ("end_index" in b_dic) else None
            pDic["ring_length"] = euclidean_distance(b_dic["ring"], b_dic["end_ring"], args["scale"]) if ("ring" in b_dic) and ("end_ring" in b_dic) else None
            pDic["hand_width"]    = euclidean_distance(b_list[4], b_list[5], args["scale"])
            pDic["index1"] = pDic["hand_width"] * (100 / pDic["hand_length"])
            pDic["max_breadth"] = euclidean_distance(b_list[7], b_list[8], args["scale"])
            pDic["hand_span"] = euclidean_distance(b_list[2], b_list[3], args["scale"])
        
    pDic['index'] = file.split("-")[0].strip(".png").strip(".jpg").strip(".ply")
    pDic["volume"] = None
    pDic["index2"] = None
    pDic["max_circumference"] = None
    pDic["max_diameter"] = None
    pDic["wrist_ratio"] = None
    pDic["hLength_bHeight"] = None
    pDic["hVolume_BMI"] = None
    pDic["bug"] = bug
    pDic["check"] = check

    if bug or b_list == None:
        pDic["hand_length"]   = None
        pDic["palm_length"]   = None
        pDic["finger_length"] = None
        pDic["hand_width"]    = None
        pDic["index1"] = None
        pDic["max_breadth"] = None

    df = df.append(pDic, ignore_index=True)
    
    del sk  # delete the skeletone object

# convert the values stored in pDic dictionary to a CSV output file.
df.to_csv(f"{output_dir}/results.csv")

with open(f"{output_dir}/results.json", "w") as outfile:
    json.dump(points_dict, outfile, indent=4)