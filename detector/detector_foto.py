import numpy as np
import cv2
from cv2.cv import *
'''
desarrollador: Brahian Velazquez Tellez 
requerimientos : Python 2.7
				Opencv
				Numpy
'''
# cargar archivo xml donde estan los algoritmos de reconocimiento
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
print("inicia el programa")
#cargamos la imagen 
img = cv2.imread('image.png')
gris = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
caras = face_cascade.detectMultiScale(gris , 1.3 ,5)
#buscamos todas las caras de la imagen
for (x,y,w,h) in caras:
	cv2.rectangle(img,(x,y),(x + w , y + h),(125,255,0), 2)


cv2.imshow('Rostro encontrado', img)	
print("numero de caras detectadas  %s") % len(caras)
cv2.imwrite('rostros_encontrados.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
 