# -*- coding: utf-8 -*-

# Facial landmarks detector - https://github.com/alexcamargoweb/facial-landmarks
# Detecção de partes de um rosto (olhos, sobrancelhas, nariz, boca e mandíbula) com Python, OpenCV e dlib.
# Referência: Adrian Rosebrock, Real-time facial landmark detection with OpenCV, Python, and dlib. PyImageSearch.
# Disponível em: https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/.
# Acessado em: 18/01/2021.
# Arquivo: facial_landmarks.py
# Execução via PyCharm/Linux (Anaconda Environment)

# importa os pacotes necessários
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2

shape_predictor = 'predictor/shape_predictor_68_face_landmarks.dat'  # preditor das partes do rosto

# inicializa o detector de rosto dlib (baseado em HOG)
detector = dlib.get_frontal_face_detector()
# em seguida, cria o preditor das partes do rosto
predictor = dlib.shape_predictor(shape_predictor)

# inicializa o stream de vídeo
vs = VideoStream(0).start()
time.sleep(2.0)

# faz um loop sobre os frames do vídeo
while True: # "q" para sair
    # carrega o frame, redimensiona e converte para uma escala de cinza
    frame = vs.read()
    frame = imutils.resize(frame, width = 500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detecta as faces no frame em escala de cinza
    rects = detector(gray, 0)

    # faz um loop nos rostos detectados
    for rect in rects:
        # determina os pontos de referência faciais para a região do rosto
        shape = predictor(gray, rect)
        # converte as coordenadas (x, y) do ponto de referência facial em uma matriz NumPy
        shape = face_utils.shape_to_np(shape)

        # faz um loop sobre as coordenadas (x, y) para os pontos de referência faciais
        for (x, y) in shape:
            # desenha os pontos na imagem
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # exibe a saída com as detecções
    cv2.imshow("Facial Landmarks video detector", frame)
    key = cv2.waitKey(1) & 0xFF

    # se a tecla 'q' for pressionada, sai do loop e finaliza a detecção
    if key == ord("q"):
        break
