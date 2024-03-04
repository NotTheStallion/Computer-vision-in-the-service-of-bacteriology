import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def detection(X,Y,seuillage=0.6,weight=0.5):
  image_globale = cv2.imread(Y, cv2.IMREAD_UNCHANGED)
  image = cv2.imread(X, cv2.IMREAD_UNCHANGED)
  result = cv2.matchTemplate(image_globale, image, cv2.TM_SQDIFF_NORMED)
  positions = np.where(result >= seuillage)
  positions1 = []
  for i in range(len(positions[0])):
     positions1.append((positions[1][i], positions[0][i]))
  largeur_image = image.shape[1]
  hauteur_image = image.shape[0]
  rectangles = []
  for pos in positions1:
    rect = [int(pos[0]), int(pos[1]), largeur_image, hauteur_image]
    rectangles.append(rect)
    rectangles.append(rect)
  rectangles, weights = cv2.groupRectangles(rectangles,1, weight)
  points = []
  for (x, y, w, h) in rectangles:
    points.append((x, y))
    cv2.drawMarker(image_globale, (x, y),(255, 0, 255), 2,40, 2)
  cv2.waitKey()
  return len(points)
x=np.linspace(0.6,1,20)
for y in x:
  print(detection("baton_hauriz.png", "Groupe_baton.png",y))

plt.show()
