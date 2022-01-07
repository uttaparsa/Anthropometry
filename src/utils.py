import math
import os
import sys
import shutil

from matplotlib.pyplot import sca

def euclidean_distance(p1, p2, scale):
    if p1 != None and p2 != None:
        return abs(math.sqrt(  ( (p1[0] - p2[0]) ** 2) + ( (p1[1] - p2[1]) ** 2) )  ) * scale
    else:
        return None


def directory_handler(args, input_path, output_path, cache_path):
    if not os.path.isdir(input_path):
        print(
            '\033[31m'
            + 'No such a directory with path: '
            + input_path
            + '\033[0m'
        )

    if not os.path.isdir(cache_path):
        os.mkdir(cache_path)
        os.mkdir(cache_path + '/fix')
        os.mkdir(cache_path + '/images')

    else:
        if args["ply"]:
            print(
                "\033[93m'" 
                + "Warning: The cache directory was not empty. All files in the directory were removed!" 
                + "\033[0m'")
            
            for filename in os.listdir(cache_path):
                file_path = os.path.join(cache_path, filename)

                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)

                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)

                except OSError as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
            
            os.mkdir("./cache/fix")
            os.mkdir("./cache/images")

    if not os.path.isdir(output_path):
        os.mkdir(output_path)


def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__