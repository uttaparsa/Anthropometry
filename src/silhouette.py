import cv2


def silhouetter(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    color_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return color_thresh