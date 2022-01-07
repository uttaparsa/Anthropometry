import argparse
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
from utils import directory_handler, euclidean_distance, blockPrint, enablePrint

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

ap.add_argument("-s", "--scale", type=float, required=True,
	help="Scale of the measurements")

ap.add_argument("-i", "--input", type=str, required=True,
	help="path to the directory of input images or ply files")

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
                        "index1",
                        "index2",
                        "max_breadth",
                        "max_circumference",
                        "max_diameter",
                        "wrist_circumference",
                        "wrist_ratio",
                        "hLength_bHeight",
                        "hVolume_BMI"
                        "hand_span"
                    ])

img_dir = input_dir

if args['ply']:
    img_dir = cache_image

    plyFiles = next(os.walk(input_dir), (None, None, []))[2]

    # walks through the given PLY files and generates fixed PLY ones in the ./cache/fix directory
    print("Generating fixed ply files ...")
    # blockPrint()
    for pFile in plyFiles:
        mesh_path = f'{input_dir}/{pFile}'
        plyFix(mesh_path, filename=pFile, dir=cache_fix)

    # enablePrint()

    # convert(project) the ply files into 2D images
    fixed_filenames = next(os.walk(cache_fix), (None, None, []))[2]  # [] if no file

    # walk through the fixed PLY files and generates the 2D image files in ./cache/image directory
    for fixed_file in fixed_filenames:
        project(img_path=f"{cache_fix}/{fixed_file}", output_path=cache_image)

# walk through the hand image files and find their points
filenames = next(os.walk(img_dir), (None, None, []))[2]  # [] if no file

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

    points = sk.keypoints()

    b_dic = b_list = None

    if points != None:
        if  points[0]  != None and \
            points[2]  != None and \
            points[4]  != None and \
            points[5]  != None and \
            points[9]  != None and \
            points[12] != None and \
            points[13] != None :

            b_dic, b_list = sk.border_points(points=points)

        elif points[20]:
            bug = True
    else:
        bug = True
    
    if not bug:
        b_img = sk.draw_border_points()
        
        if b_img is not None:
            cv2.imwrite(f"{output_dir}/{file.strip('.ply').strip('.jpg').strip('.png')}.jpg", b_img)

        if b_list != None:
            pDic["hand_length"]   = euclidean_distance(b_list[0], b_list[1], args["scale"])
            pDic["palm_length"]   = euclidean_distance(b_list[1], b_list[6], args["scale"])
            pDic["finger_length"] = euclidean_distance(b_list[0], b_list[6], args["scale"])
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
