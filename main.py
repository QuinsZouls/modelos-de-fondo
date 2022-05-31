import numpy as np
import cv2
import time
from collections import deque

# Cargamos el vídeo
camara = cv2.VideoCapture("video.mp4")

# Inicializamos el primer frame a vacío.
# Nos servirá para obtener el fondo
fondo = None
points = []
# Recorremos todos los frames
while True:
	# Obtenemos el frame
	(grabbed, frame) = camara.read()

	# Si hemos llegado al final del vídeo salimos
	if not grabbed:
		break

	# Convertimos a escala de grises
	gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Aplicamos suavizado para eliminar ruido
	gris = cv2.GaussianBlur(gris, (21, 21), 0)

	# Si todavía no hemos obtenido el fondo, lo obtenemos
	# Será el primer frame que obtengamos
	if fondo is None:
		fondo = gris
		height, width = frame.shape[:2]
		accum_image = np.zeros((height, width), np.uint8)
		continue

	# Calculo de la diferencia entre el fondo y el frame actual
	resta = cv2.absdiff(fondo, gris)

	# Aplicamos un umbral
	umbral = cv2.threshold(resta, 25, 255, cv2.THRESH_BINARY)[1]

	# Dilatamos el umbral para tapar agujeros
	umbral = cv2.dilate(umbral, None, iterations=2)

	# Copiamos el umbral para detectar los contornos
	contornosimg = umbral.copy()

	# Buscamos contorno en la imagen
	contornos, hierarchy = cv2.findContours(contornosimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	center = None
	# Recorremos todos los contornos encontrados
	for c in contornos:
		# Eliminamos los contornos más pequeños
		if cv2.contourArea(c) < 500:
			continue

		# Obtenemos el bounds del contorno, el rectángulo mayor que engloba al contorno
		(x, y, w, h) = cv2.boundingRect(c)
		#Agregamos los puntos al historial
		points.append([x, y + h])

	#Mostramos el historial de puntos

	# Creamos un heatmap con el fondo que tenemos
	threshold = 1
	maxValue = 1
	ret, th1 = cv2.threshold(umbral, threshold, maxValue, cv2.THRESH_BINARY)
	accum_image = cv2.add(accum_image, th1)
	color_image_video = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
	video_frame = cv2.addWeighted(frame, 0.7, color_image_video, 0.7, 0)

	cv2.imshow('Resta',video_frame)
	# Capturamos una tecla para salir
	key = cv2.waitKey(1) & 0xFF

	# Tiempo de espera para que se vea bien
	time.sleep(0.015)

	# Si ha pulsado la letra s, salimos
	if key == ord("s"):
		break

# Liberamos la cámara y cerramos todas las ventanas
camara.release()
cv2.destroyAllWindows()