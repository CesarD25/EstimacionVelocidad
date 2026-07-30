import cv2
import numpy as np


def dibujar_lineas(frame, a1, a2, b1, b2):
    cv2.line(frame, tuple(a1), tuple(a2), (255, 0, 0), 2)
    cv2.putText(frame, 'A', (a1[0] + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.line(frame, tuple(b1), tuple(b2), (0, 0, 255), 2)
    cv2.putText(frame, 'B', (b1[0] + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def transformar_punto(punto, homografia):
    punto_homogeneo = np.array([punto[0], punto[1], 1.0])
    resultado = np.asarray(homografia) @ punto_homogeneo
    resultado /= resultado[2]
    return float(resultado[0]), float(resultado[1])
