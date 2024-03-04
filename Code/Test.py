import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Zoom(image,facteur):
   x = int(image.shape[0] * facteur)
   y = int(image.shape[1] * facteur)
   return cv2.resize(image, (x, y))

def Contours(img, valeur_seuillage=160):
   image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
   image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   image_black = cv2.threshold(image_gray, valeur_seuillage, 255, cv2.THRESH_BINARY)[1]
   contours = cv2.findContours(image_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
   #cv2.drawContours(image, contours, 1, (0, 255, 0), 1)
   #cv2.imshow("image apres contour",Zoom(image,20))
   #Y=[]
   #X=[]
   #for x in contours[1]:
      #X.append(x[0][0])
      #Y.append(x[0][1])
   #print(Y)
   #plt.plot(X,Y,'ro')
   #plt.show()
   return contours

def Pentes(contour, pas=1):
   L = np.ones(0)
   X=[]
   Y=[]
   for i in range(len(contour)):
      X.append(contour[i][0][0])
      Y.append(contour[i][0][1])
   for i in range(0, len(contour), pas):
      if contour[i][0][0] == contour[i - 1][0][0]:
         if contour[i][0][1] - contour[i - 1][0][1]>0:
            L = np.append(L, [500])
         else:
            L = np.append(L, [-500])
      else:
         L = np.append(L, [(contour[i][0][1] - contour[i - 1][0][1]) / (contour[i][0][0] - contour[i - 1][0][0])])
   return L

plt.plot(Pentes(Contours("baton_hauriz.png")[1]),'b')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()