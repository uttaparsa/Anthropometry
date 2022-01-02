from __future__ import division
import cv2
import time
import numpy as np
from numpy.core.numeric import count_nonzero
from numpy.lib.function_base import select


class Skeletone:
    image: np.array
    image_height: int
    image_width:  int

    proto_file = "model/pose_deploy.prototxt"
    weight_file = "model/pose_iter_102000.caffemodel"
    
    POSE_PAIRS = [ 
                [0, 1], [1, 2], [2, 3], [3, 4], 
                [0, 5], [5, 6], [6, 7], [7, 8], 
                [0, 9], [9, 10], [10, 11], [11, 12], 
                [0, 13], [13, 14], [14, 15], [15, 16], 
                [0, 17], [17, 18], [18, 19], [19, 20] 
                ]
    nPoints = 22
    border_points_list: list
    points: list


    def __init__(self, image) -> None:
        self.image = image
        self.image_height = image.shape[0]
        self.image_width  = image.shape[1]
        self.points = []
        self.border_points_list = []


    def keypoints(self):
        net = cv2.dnn.readNetFromCaffe(self.proto_file, self.weight_file)

        imgCopy = np.copy(self.image)

        aspect_ratio = self.image_width / self.image_height
        threshold = 0.1

        # input image dimensions for the network
        inHeight = 368
        inWidth = int(((aspect_ratio * inHeight) * 8) // 8)
        inpBlob = cv2.dnn.blobFromImage(self.image, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

        net.setInput(inpBlob)

        output = net.forward()

        # Empty list to store the detected keypoints
        for i in range(self.nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]
            probMap = cv2.resize(probMap, (self.image_width, self.image_height))

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            if prob > threshold :
                cv2.circle(imgCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(imgCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                self.points.append((int(point[0]), int(point[1])))
            else :
                self.points.append(None)
        return self.points


    def draw_skeletone(self, points: list):
        if len(points) > 0:
            for pair in self.POSE_PAIRS:
                partA = pair[0]
                partB = pair[1]

                if points[partA] and points[partB]:
                    cv2.line(self.image, points[partA], points[partB], (0, 255, 255), 2)
                    cv2.circle(self.image, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.circle(self.image, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    

    def border_points(self, points) -> tuple:
        point2  = points[2]
        thumb   = points[4]
        point5  = points[5]
        point9  = points[9]
        point12 = points[12]
        point17 = points[17]

        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        border_point_dict = dict()

        # middle finger top point
        currentY = point12[1]
        while gray_image[currentY, point12[0]] != 0:
            currentY += 1
        
        new_top = [point12[0], currentY]
        border_point_dict["top"] = new_top
        self.border_points_list.append(new_top)

        # --------------------------------------------
        border_point_dict["wrist"] = points[0]
        self.border_points_list.append(points[0])
        border_point_dict["thumb"] = thumb
        self.border_points_list.append(thumb)
        border_point_dict["little"] = points[20]
        self.border_points_list.append(points[20])

        # --------------------------------------------
        # hand inner border points
        currentX = point17[0]
        
        if point17[0] > thumb[0]:
            while gray_image[point17[1], currentX] != 0:
                currentX += 1
        else:
            while gray_image[point17[1], currentX] != 0:
                currentX -= 1
        
        side1 = [currentX, point17[1]]
        border_point_dict["side1"] = side1
        self.border_points_list.append(side1)
# ----------------------------------------------------------------------------
        # hand inner border points (inner)
        # find the side point, which is between thumb finger and index finger
        currentX = point5[0]
        if point17[0] > thumb[0]:
            while gray_image[point5[1], currentX] != 0:
                currentX -= 1
        else:
            while gray_image[point5[1], currentX] != 0:
                currentX += 1        

        side2 = [currentX, point5[1]]
        border_point_dict["side2"] = side2
        self.border_points_list.append(side2)

     # ----------------------------------------------------------   
        # find the end point of the middle finger

        middle3X = int(np.floor((point5[0] + point9[0]) / 2))
        middle2X = int(np.floor((point5[0] + middle3X)  / 2))
        middle4X = int(np.floor((middle3X  + point9[0]) / 2))
        middle5X = int(np.floor((middle4X  + point9[0]) / 2))
        middle1X = int(np.floor((point5[0] + middle2X)  / 2))
        middleY  = point5[1]

        black_alert = False
        
        print("initial y: ", middleY)

        while black_alert == False:
            if gray_image[middleY, middle1X] == 0:
                black_alert = True
                break

            if gray_image[middleY, middle2X] == 0:
                black_alert = True
                break
            
            if gray_image[middleY, middle3X] == 0:
                black_alert = True
                break
            
            if gray_image[middleY, middle4X] == 0:
                black_alert = True
                break

            if gray_image[middleY, middle5X] == 0:
                black_alert = True
                break
            
            middleY += 1
        
        end_middle = [points[12][0], middleY]
        border_point_dict['end_middle'] = end_middle
        self.border_points_list.append(end_middle)
# -----------------------------------------------------------
        # find max width of the hand (border ponts of the hands 7-8)
        currentX1 = point2[0]
        while gray_image[point2[1], currentX1] != 0:
            currentX1 -= 1
        
        side3 = [currentX1, point2[1]]
        border_point_dict["side4"] = side3
        self.border_points_list.append(side3)

        currentX2 = point2[0]
        while gray_image[point2[1], currentX2] != 0:
            currentX2 += 1
        
        side4 = [currentX2, point2[1]]
        border_point_dict["side4"] = side4
        self.border_points_list.append(side4)
# -----------------------------------------------------------
        # find points around wrist (9-10)
        wrist = points[0]
        currentX1 = wrist[0]
        while gray_image[wrist[1], currentX1] != 0:
            currentX1 -= 1
        
        side5 = [currentX1, wrist[1]]
        border_point_dict["side5"] = side5  
        self.border_points_list.append(side5)

        currentX2 = wrist[0]
        while gray_image[wrist[1], currentX2] != 0:
            currentX2 += 1
        
        side6 = [currentX2, wrist[1]]
        border_point_dict["side6"] = side6
        self.border_points_list.append(side6)
# ----------------------------------------------------------
        return border_point_dict, self.border_points_list

    
    def draw_border_points(self):
        imgCopy = np.copy(self.image)

        if len(self.border_points_list) > 0:
            for i in range(len(self.border_points_list)):
                cv2.circle(imgCopy, (int(self.border_points_list[i][0]), int(self.border_points_list[i][1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(imgCopy, "{}".format(i), (int(self.border_points_list[i][0]), int(self.border_points_list[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        else:
            self.border_points()

            if len(self.border_points) > 0:
                for i in range(len(self.border_points_list)):
                    cv2.circle(imgCopy, (int(self.border_points_list[i][0]), int(self.border_points_list[i][1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.putText(imgCopy, "{}".format(i), (int(self.border_points_list[i][0]), int(self.border_points_list[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        return imgCopy

    
    def draw_points(self, pList):
        imgCopy = np.copy(self.image)

        if len(pList) > 0:
            for i in range(len(pList)):
                cv2.circle(imgCopy, (int(pList[i][0]), int(pList[i][1])), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(imgCopy, "{}".format(i), (int(pList[i][0]), int(pList[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.imshow("points", imgCopy)
        cv2.waitKey(0)

    
    def get_hand_start(self):
        wrist = self.points[0]
        currentY1 = wrist[1]

        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        while gray_image[wrist[1], currentY1] != 0:
            currentY1 -= 1

        print("the beginning point of the hand is: ({}, {})".format(wrist[0], currentY1))


    def get_top_bottom(self):
        """
        returns the highest and lowest point of the hand
        """
        top = ()
        bottom = ()
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        top_found = False
        bottom_found = False

        for i in range(gray.shape[0]):
            if not top_found:
                for j in range(gray.shape[1]):
                    # print("i, j =", (i, j))
                    if gray[i, j] != 0:
                        top = (i, j)
                        top_found = True
                        break
            
        for i in range(gray.shape[0], 0, -1):
            if not bottom_found:
                for j in range(gray.shape[1]):
                    # print("^^ i, j =", (i, j))
                    if gray[i - 1, j - 1] == 255:
                        bottom = (i - 1, j - 1)
                        bottom_found = True
                        break
                    
        return top, bottom
