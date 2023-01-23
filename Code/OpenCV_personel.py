
import cv2
import numpy as np
seuillage = 0.7
image_globale = cv2.imread('Groupe_rond.png', cv2.IMREAD_UNCHANGED)
image = cv2.imread('Rond_1.png', cv2.IMREAD_UNCHANGED)
# cv2.matchTemplate permet de comparer deux images et renvoi un tableau
# qui contient les valeur de degrée de resemblance (=image où chaque
# pixel contient [0,1] plus c'est blanc (=1) plus ça resemble)
result = cv2.matchTemplate(image_globale, image, cv2.TM_SQDIFF_NORMED)
cv2.imshow('resultat',result)
# np.where retourn toute les positions des pixels où le degrée de
# resemblance est supérieure à la valeur seuillage
positions = np.where(result >= seuillage)
# La particularité de cette methode c'est qu'elle rend une liste qui contient
# une liste des abssices et une liste des ordonnées [[y1...];[x1...]]
position=[]
for i in range(len(positions[0])):
    position.append((positions[1][i], positions[0][i]))
# Transforme la liste de listes en liste de couple (x,y)
for pos in position:
    cv2.drawMarker(image_globale, pos, (0, 255, 0), 2)
#cv2.imshow('Matches', image_globale)

def SupprimerRep(positions,e):
    L=[]
    for i in range(len(positions)):
        if ((positions[i][1]-positions[i-1][1])**2)+((positions[i][1]-positions[i-1][1])**2)>e:
            L.append(positions[i])
    for l in L:
        cv2.drawMarker(image_globale, l, (0, 255, 0), 10, None,5)
    return L

print(len(SupprimerRep(position, 1.9)))
print(len(position))

cv2.imshow('Matches', image_globale)
cv2.waitKey(0)
cv2.destroyAllWindows()
