import numpy as np
import cv2 as cv

# Cargar el clasificador de rostros
clasificador_rostro = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Inicializar la captura de video
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

# Contador para guardar imágenes (opcional)
contador = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el cuadro.")
        break

    # Convertir el marco a escala de grises
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detectar rostros en el marco
    rostros = clasificador_rostro.detectMultiScale(frame_gray, 1.3, 5)

    for (x, y, w, h) in rostros:
        # Dibujar un rectángulo alrededor del rostro detectado
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Recortar y redimensionar el rostro detectado
        rostro_recortado = frame[y:y + h, x:x + w]
        rostro_recortado = cv.resize(rostro_recortado, (80, 80), interpolation=cv.INTER_AREA)

        # Convertir el rostro recortado a escala de grises
        rostro_gris = cv.cvtColor(rostro_recortado, cv.COLOR_BGR2GRAY)

        # Mostrar el rostro recortado y su versión en escala de grises
        cv.imshow('Rostro Detectado', rostro_recortado)
        cv.imshow('Rostro en Escala de Grises', rostro_gris)

    # Mostrar el marco original con los rectángulos
    cv.imshow('Detección de Rostros', frame)

    contador += 1

    # Salir del bucle al presionar 'Esc'
    if cv.waitKey(1) & 0xFF == 27:
        break

# Liberar recursos
cap.release()
cv.destroyAllWindows()
