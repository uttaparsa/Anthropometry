
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
                        "filename",
                        "top",
                        "bottom",
                        "wrist","thumb","little","side1","side2","side4","side5","side6","end_middle"]
                        )


# run code for all the images
# filenames = next(walk(data_dir), (None, None, []))[2]  # [] if no file
# print("file names:  ", filenames)

# for file in filenames:
#     mesh_path = f'{data_dir}/{file}'
#     restoration(mesh_path, filename=file, dir=fixed_data_dir)

filenames = next(walk(fixed_data_dir), (None, None, []))[2]  # [] if no file
print("file names:  ", filenames)

done_count = 0

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


        print(f"b_dic {b_dic}")

        print("top2 : ", list(top))
        print("bottom: ", list(bottom))

        pDic["top"] = list(top)
        pDic["bottom"] = list(bottom)
        pDic["wrist"] = b_dic['wrist']
        pDic["thumb"] = b_dic['thumb']
        pDic["little"] = b_dic['little']
        pDic["side6"] = b_dic['side6']
        pDic["side5"] = b_dic['side5']
        pDic["side1"] = b_dic['side1']
        pDic["side2"] = b_dic['side2']
        pDic["side4"] = b_dic['side4']
        pDic["end_middle"] = b_dic['end_middle']
        

        done_count += 1
        # if done_count > 2:
        #     break
    except Exception as e:
        print(f"exception: {e}")

    pDic["filename"] = file.strip('.ply')
    df = df.append(pDic, ignore_index=True)



print("dataframe: ", df)

df.to_csv(f"{output_dir}/keypoints.csv")