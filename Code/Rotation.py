import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def rotation(image,teta):
    maxl = max(len(image),len(image[0]))
    img=[[255]*maxl*5 for i in range(maxl*5)]

    for i in range(maxl-1):
        for j in range(maxl-1):
            x = 0
            y = 0
            x=int(i*np.cos(teta)+j*np.sin(teta))
            y=int(-i*np.sin(teta)+j*np.cos(teta))

            try:
                img[i][j] = image[x][y]
            except:
                pass
    return img

image = cv2.imread("test.png", cv2.IMREAD_UNCHANGED)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_black = cv2.threshold(image_gray, 160, 255, cv2.THRESH_BINARY)[1]
x=rotation(image_black,20)
print(x)
cv2.imshow("rot",np.array(x))
