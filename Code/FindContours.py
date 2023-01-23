import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def Contours(img, valeur_seuillage=160):
   image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
   image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   image_black = cv2.threshold(image_gray, valeur_seuillage, 255, cv2.THRESH_BINARY)[1]
   contours = cv2.findContours(image_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
   #cv2.drawContours(image, contours, 1, (0, 255, 0), 3)
   #cv2.imshow("image apres contour",image)
   #cv2.waitKey(0)
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

def MoyGlissante(x, n):
   if len(x) < n:
      return str(n) + "est une valeur invalide"
   L = np.ones(0)
   s = 0
   for i in range(len(x)):
      if i < n - 1:                       # i < n-1
         for j in range(i + 1):
            s += x[j]                     # calcule de la somme des élements avant i
         L = np.append(L, s / (i + 1))    # affectation de la nouvelle valeur
         s = 0                            # réinitialisation de la variable somme#
      else:                               # i > n-1
         for j in range(i - n - 1, i + 1):
            s += x[j]                     # calcule de la somme de n élements avant i
         L = np.append(L, s / n)          # affectation de la nouvelle valeur
         s = 0                            # réinitialisation de la variable somme
   return L

def diff(L, H):
   S = 0
   for i in range(len(L)):
      S += abs(L[i] - H[i])
   return S / len(L)

def detection_rond(img,valeur_seuillage=160,Moy=50):
   contours = Contours(img,valeur_seuillage)
   rond=0
   Y1=np.ones(0)
   print("Le nombre de contours trouvé est : ",len(contours))

   for contour in contours :
      Y1 = MoyGlissante(Pentes(Contours("rond_1.png", valeur_seuillage)[1]), Moy)

      Y3= MoyGlissante(Pentes(contour), Moy)
      X1 = np.arange(len(Y1))

      X3 = np.arange(len(Y3))
      y3 = interp1d(X3, Y3, kind="linear")
      y1 = interp1d(X1, Y1, kind="linear")

      l = max(len(Y1), len(Y3))
      X1 = np.linspace(0, len(Y1) - 1, l)

      X3 = np.linspace(0, len(Y3) - 1, l)
      new1 = np.array([y1(x) for x in X1])

      new3 = np.array([y3(x) for x in X3])
      #print("rond ",diff(new3, new1),)
      if diff(new3,new1)<125:
         rond+=1

   print("Le nombre de cocci dans l'image donnée est : ",rond)
   return rond

def detection_baton(img,valeur_seuillage=160,Moy=50):
   contours = Contours(img,valeur_seuillage)
   baton=0
   Y1=np.ones(0)
   print("Le nombre de contours trouvé est : ",len(contours))

   for contour in contours :
      Y1 = MoyGlissante(Pentes(Contours("baton_hauriz.png", valeur_seuillage)[1]), Moy)

      Y3= MoyGlissante(Pentes(contour), Moy)
      X1 = np.arange(len(Y1))

      X3 = np.arange(len(Y3))
      y3 = interp1d(X3, Y3, kind="linear")
      y1 = interp1d(X1, Y1, kind="linear")

      l = max(len(Y1), len(Y3))
      X1 = np.linspace(0, len(Y1) - 1, l)

      X3 = np.linspace(0, len(Y3) - 1, l)
      new1 = np.array([y1(x) for x in X1])

      new3 = np.array([y3(x) for x in X3])
      #print("baton ",diff(new3, new1),)
      if diff(new3,new1)<100:
         baton+=1

   print("Le nombre de baciles dans l'image donnée est : ", baton)
   return baton


detection_rond("Groupe_rond.png")
detection_baton("Groupe_rond.png")