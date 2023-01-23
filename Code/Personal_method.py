import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def Zoom(image,facteur):
   x = int(image.shape[0] * facteur)
   y = int(image.shape[1] * facteur)
   return cv2.resize(image, (x, y))

def Contours(img, valeur_seuillage=160):
   image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
   image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   image_black = cv2.threshold(image_gray, valeur_seuillage, 255, cv2.THRESH_BINARY)[1]
   contours = cv2.findContours(image_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
   #cv2.drawContours(image, contours, 1, (0, 255, 0), 3)
   #cv2.imshow("image apres contour",Zoom(image,20))
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

#Pentes(Contours("cherch.png",160)[1],1)
Y1=MoyGlissante(Pentes(Contours("rond.png",)[1]),50)
Y2=MoyGlissante(Pentes(Contours("rond_1.png",)[1]),50)
Y3=MoyGlissante(Pentes(Contours("Rond_1.png",)[1]),50)
#Y1=Pentes(Contours("rond.png",)[1])
#Y2=Pentes(Contours("rond_defo.png",)[1])
#Y3=Pentes(Contours("rond_blur_light_texture.png",)[1])
print(len(Y1),len(Y2),len(Y3))
X1=np.arange(len(Y1))
X2=np.arange(len(Y2))
X3=np.arange(len(Y3))

y1=interp1d(X1,Y1,kind="linear")
y2=interp1d(X2,Y2,kind="linear")
y3=interp1d(X3,Y3,kind="linear")

l=max(len(Y1),len(Y2),len(Y3))

X1=np.linspace(0,len(Y1)-1,l)
X2=np.linspace(0,len(Y2)-1,l)
X3=np.linspace(0,len(Y3)-1,l)

new1=np.array([y1(x) for x in X1])
new2=np.array([y2(x) for x in X2])
new3=np.array([y3(x) for x in X3])

print(diff(new3,new1),diff(new2,new3),diff(new1,new2))

X1=np.arange(0,l,l/len(Y1))
X2=np.arange(0,l,l/len(Y2))
X3=np.arange(0,l,l/len(Y3))

plt.plot(X1,Y1,"r")
plt.plot(X2,Y2,'b')
plt.plot(X3,Y3,'y')


#image = cv2.imread("Pixel21.png", cv2.IMREAD_UNCHANGED)
#image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#image_black = cv2.threshold(image_gray, 160, 255, cv2.THRESH_BINARY)[1]
#cv2.imshow("image_black",Zoom(image_black,20))
# La fonction findcontours trouve tout les contour present dans l'image inclus le cadre
#contours = cv2.findContours(image_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
# La fonction drawcontours permet de tracer tout ces contours
#cv2.drawContours(image, contours, 1, (0, 255, 0), 1)
# visualiser l'image d'origine avec tout les contours
#cv2.imshow("image apres contour et zoom",Zoom(image,20))

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()