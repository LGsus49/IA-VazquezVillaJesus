import numpy as np
import cv2 as cv

# Cargar el clasificador de rostros
clasificador_rostro = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Iniciar captura de video
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

# Contador para las imágenes guardadas
contador = 0

# Cantidad previa de píxeles blancos
blancos_previos = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el cuadro.")
        break

    # Convertir a escala de grises
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detectar rostros
    rostros = clasificador_rostro.detectMultiScale(frame_gray, 1.3, 5)

    for (x, y, w, h) in rostros:
        # Recortar el rostro detectado
        rostro_recortado = frame[y:y + h, x:x + w]
        rostro_recortado = cv.resize(rostro_recortado, (100, 100), interpolation=cv.INTER_AREA)

        # Guardar el rostro recortado
        cv.imwrite(f'capturas/Face_{contador}.jpg', rostro_recortado)

        # Convertir el rostro a escala de grises
        rostro_gris = cv.cvtColor(rostro_recortado, cv.COLOR_BGR2GRAY)

        # Convertir a binario (blanco y negro)
        _, rostro_binario = cv.threshold(rostro_gris, 127, 255, cv.THRESH_BINARY)

        # Contar los píxeles blancos
        blancos_actuales = cv.countNonZero(rostro_binario)
        diferencia_blancos = blancos_actuales - blancos_previos

        # Mostrar información en consola
        print(f"Blancos actuales: {blancos_actuales}, Diferencia: {diferencia_blancos}")

        # Actualizar los píxeles blancos previos
        blancos_previos = blancos_actuales

        # Mostrar los resultados
        cv.imshow('Rostro Detectado', rostro_recortado)
        cv.imshow('Rostro Binario', rostro_binario)

    # Mostrar el marco original
    cv.imshow('Detección de Rostros', frame)

    contador += 1

    # Salir al presionar 'Esc'
    if cv.waitKey(1) & 0xFF == 27:
        break

# Liberar recursos
cap.release()
cv.destroyAllWindows()
