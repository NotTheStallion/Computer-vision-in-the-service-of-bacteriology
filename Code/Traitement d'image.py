
# Il faut importer la bibliothèque
import cv2

# La fonction imread permet de lire l'image stockée et la transformer en tableau de
# valeur de pixel.
# (imread_color,imread_grayscale,imread_unchanged)
image = cv2.imread("Pixel21.png", cv2.IMREAD_UNCHANGED)
# La fonction cvtColor permet de convertir l'image en gris
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Seuiller (=threshold) l'image revient à transformer l'image en image binaire.
Valeur_seuillage=160
image_black=cv2.threshold(image_gray, Valeur_seuillage, 255, cv2.THRESH_BINARY)[1]
# La fonction imshow("nom de l'image affichée",image)
# cv2.imshow("image en noir",image_black)

#print(img)
print(image_black)