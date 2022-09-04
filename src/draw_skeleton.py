"""
A code to run on the pourreza's images
"""
import cv2
from os import walk
from skeletone import Skeletone



data_dir = "./Data"
fixed_data_dir = "./Data/fixed"
output_dir = "./points_output"

p_dir = "./cache/pour2"

# The following line is just for in the case of having the images projected with the Matlab code
filenames = next(walk(p_dir), (None, None, []))[2]  # [] if no file

for file in filenames:
    print("file: ", file)
    # The following two lines are just for in the case of having the images projected with the Matlab code
    sil_img = cv2.imread(f"{p_dir}/{file}")

    sk = Skeletone(image=sil_img, name=file.split(".")[0])
    bug = False

    points = sk.keypoints()
    kp_img = sk.draw_points(points)

    if kp_img is not None:
        cv2.imwrite(f"{output_dir}/{file}", kp_img)

    
    del sk