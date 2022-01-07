# ------------------------- BODY, FACE AND HAND MODELS -------------------------
# Downloading body pose (COCO and MPI), face and hand models
OPENPOSE_URL="http://posefs1.perception.cs.cmu.edu/OpenPose/models/"
MODEL_FOLDER="model/"

# "------------------------- HAND MODELS -------------------------"
# Hand
HAND_MODEL=$MODEL_FOLDER"pose_iter_102000.caffemodel"
wget -c ${OPENPOSE_URL}${HAND_MODEL} -P ${MODEL_FOLDER}