import cv2


def background(img):
    """
    This function find the background color in a silhouette image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height = gray.shape[0]
    width  = gray.shape[1]

    black = 0
    white = 0

    p0 = gray[5, 5]
    p1 = gray[5, width - 5]
    p2 = gray[height - 5, 5]
    p3 = gray[height - 5, width - 5]

    if p0 == 0:
        black += 1
    else:
        white += 1

    if p1 == 0:
        black += 1
    else:
        white += 1

    if p2 == 0:
        black += 1
    else:
        white += 1

    if p3 == 0:
        black += 1
    else:
        white += 1
    

    if black > white:
        return 0
    else:
        return 255


def silhouetter(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if background(img) == 0:
        _,thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _,thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    color_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return color_thresh