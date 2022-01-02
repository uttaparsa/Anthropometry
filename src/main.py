
import trimesh
import pandas as pd
import numpy as np
import cv2
from os import walk
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

from silhouette import silhouetter
from skeletone import Skeletone
from utils import euclidean_distance, restoration


data_dir = "./Data"
fixed_data_dir = "./Data/fixed"
output_dir = "./output"
df = pd.DataFrame(columns = [
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


# run code for all the images

# for file in filenames:
#     mesh_path = f'{data_dir}/{file}'
#     restoration(mesh_path, filename=file, dir=fixed_data_dir)

filenames = next(walk(fixed_data_dir), (None, None, []))[2]  # [] if no file
print("file names:  ", filenames)

for file in filenames:
    pDic = dict()

    mesh = trimesh.load_mesh(f'{fixed_data_dir}/{file}')
    mesh.is_watertight

    plane = trimesh.points.project_to_plane(mesh.vertices, plane_normal=[0,0,1],plane_origin=[0,0,0])


    figure(figsize=(8, 12), dpi=100)

    data = np.array(plane)
    x, y = data.T
    plt.scatter(x,y)
    plt.axis('off')
    plt.savefig(f"cache/{file.strip('.ply')}.png", bbox_inches='tight')

    # mesh.bounding_box_oriented.show()

    img = cv2.imread(f"cache/{file.strip('.ply')}.png")

    sil_img = silhouetter(img)

    sk = Skeletone(image=sil_img)

    try:
        points = sk.keypoints()
        b_dic, b_list = sk.border_points(points=points)

        b_img = sk.draw_border_points()
        cv2.imwrite(f"{output_dir}/{file.strip('.ply')}.jpg", b_img)

        top, bottom = sk.get_top_bottom()

        print("top : ", top)
        print("bottom: ", bottom)

        pDic["volume"] = None
        pDic["hand_length"]   = euclidean_distance(b_list[0], b_list[1])
        pDic["palm_length"]   = euclidean_distance(b_list[1], b_list[6])
        pDic["finger_length"] = euclidean_distance(b_list[0], b_list[6])
        pDic["hand_width"]    = euclidean_distance(b_list[4], b_list[5])
        pDic["index1"] = pDic["hand_width"] * (100 / pDic["hand_length"])
        pDic["index2"] = None
        pDic["max_breadth"] = euclidean_distance(b_list[7], b_list[8])
        pDic["max_circumference"] = None
        pDic["max_diameter"] = None
        pDic["wrist_ratio"] = None
        pDic["hLength_bHeight",] = None
        pDic["hVolume_BMI"] = None
        pDic["hand_span"] = euclidean_distance(b_list[2], b_list[3])
    
    except:
        pDic["volume"] = None
        pDic["hand_length"]   = None
        pDic["palm_length"]   = None
        pDic["finger_length"] = None
        pDic["hand_width"]    = None
        pDic["index1"] = None
        pDic["index2"] = None
        
        pDic["max_breadth"] = None
        pDic["max_circumference"] = None
        pDic["max_diameter"] = None
        pDic["wrist_ratio"] = None
        pDic["hLength_bHeight",] = None
        pDic["hVolume_BMI"] = None
        pDic["hand_span"] = None


    df = df.append(pDic, ignore_index=True)



print("dataframe: ", df)

df.to_csv(f"{output_dir}/results.csv")