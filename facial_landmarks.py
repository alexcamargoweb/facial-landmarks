# -*- coding: utf-8 -*-

# Facial landmarks detector - https://github.com/alexcamargoweb/facial-landmarks
# Detecção de partes de um rosto (olhos, sobrancelhas, nariz, boca e mandíbula) com Python, OpenCV e dlib.
# Referência: Adrian Rosebrock, Facial landmarks with dlib, OpenCV, and Python. PyImageSearch.
# Disponível em: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/.
# Acessado em: 05/01/2021.
# Arquivo: facial_landmarks.py
# Execução via PyCharm/Linux (Anaconda Environment)

# importa os pacotes necessários
from imutils import face_utils
import imutils
import dlib
import cv2

shape_predictor = 'predictor/shape_predictor_68_face_landmarks.dat' # preditor das partes do rosto
input_image = 'input/input.jpg' # imagem a ser detectada
bbox = True # exibe ou não a bouding box

# inicializa o detector de rosto dlib (baseado em HOG)
detector = dlib.get_frontal_face_detector()
# em seguida, cria o preditor das partes do rosto
predictor = dlib.shape_predictor(shape_predictor)

# carrega a imagem, redimensiona e converte para uma escala de cinza
image = cv2.imread(input_image)
image = imutils.resize(image, width = 500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detecta as faces em scala de cinza
rects = detector(gray, 1)

# faz um loop nos rostos detectados
for (i, rect) in enumerate(rects):
    # determina os pontos de referência faciais para a região do rosto
    shape = predictor(gray, rect)
    # converte as coordenadas (x, y) do ponto de referência facial em uma matriz NumPy
    shape = face_utils.shape_to_np(shape)

    # converte o retângulo dlib numa bouding box no estilo OpenCV
    (x, y, w, h) = face_utils.rect_to_bb(rect)

    # caso a opção de mostrar a bounding box esteja habilitada
    if bbox == True:
        # [i.e., (x, y, w, h)] desenha a caixa delimitadora de rosto
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # faz um loop sobre as coordenadas (x, y) para os pontos de referência faciais
    for (x, y) in shape:
        # desenha os pontos na imagem
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

# exibe a imagem de saída com as detecções
cv2.imshow("Facial Landmarks image detector", image)
cv2.waitKey(0)
