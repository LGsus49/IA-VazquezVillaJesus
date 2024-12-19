import numpy as np
import cv2 as cv

# Inicialización de variables
clasificador_rostro = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
captura = cv.VideoCapture(0)
blancos_anterior = 0
contador = 0

while True:
    ret, frame = captura.read()
    if not ret:
        print("Error al acceder a la cámara.")
        break

    # Convertir a escala de grises
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detectar rostros
    rostros = clasificador_rostro.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in rostros:
        # Procesar cada rostro detectado
        rostro_recortado = frame[y:y+h, x:x+w]
        rostro_recortado = cv.resize(rostro_recortado, (100, 100), interpolation=cv.INTER_AREA)

        # Convertir a escala de grises y aplicar binarización
        rostro_gris = cv.cvtColor(rostro_recortado, cv.COLOR_BGR2GRAY)
        _, rostro_binario = cv.threshold(rostro_gris, 127, 255, cv.THRESH_BINARY)

        # Contar píxeles blancos
        blancos_actual = cv.countNonZero(rostro_binario)

        # Comparar cantidad de píxeles blancos
        if contador > 0:
            diferencia = blancos_actual - blancos_anterior
            print(f'Blancos actuales: {blancos_actual}, Diferencia: {diferencia}')
        else:
            print(f'Blancos actuales: {blancos_actual}')

        # Actualizar blancos anteriores
        blancos_anterior = blancos_actual

        # Guardar y mostrar el rostro binarizado
        cv.imwrite(f'capturas/RostroBinario_{contador}.jpg', rostro_binario)
        cv.imshow('Rostro Binario', rostro_binario)

    # Mostrar el frame original con detección de rostros
    cv.imshow('Rostros Detectados', frame)

    # Incrementar contador y manejar salida
    contador += 1
    if cv.waitKey(1) & 0xFF == 27:  # Presiona 'Esc' para salir
        break

# Liberar recursos
captura.release()
cv.destroyAllWindows()
