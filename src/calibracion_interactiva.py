"""Calibración interactiva del plano de movimiento mediante homografía."""

import json
import os
from datetime import datetime, timezone

import cv2
import numpy as np

from src.captura import redimensionar_estandar

CAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "calibraciones")


def _seleccionar(frame, cantidad, titulo, etiquetas):
    puntos = []

    def callback(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(puntos) < cantidad:
            puntos.append((int(x), int(y)))

    cv2.namedWindow(titulo)
    cv2.setMouseCallback(titulo, callback)
    while True:
        vista = frame.copy()
        for i, punto in enumerate(puntos):
            cv2.circle(vista, punto, 6, (0, 255, 255), -1)
            cv2.putText(vista, etiquetas[i], (punto[0] + 5, punto[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vista, "Enter: confirmar | R/ESC: reiniciar",
                    (10, vista.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2)
        cv2.imshow(titulo, vista)
        tecla = cv2.waitKey(20) & 0xFF
        if tecla in (ord("r"), 27):
            puntos.clear()
        if len(puntos) == cantidad and tecla in (10, 13):
            break
    cv2.destroyWindow(titulo)
    return puntos


def _seleccionar_lineas_verticales(frame):
    posiciones_x = []
    titulo = "Calibracion - Lineas verticales A y B"

    def callback(event, x, _y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(posiciones_x) < 2:
            posiciones_x.append(int(x))

    cv2.namedWindow(titulo)
    cv2.setMouseCallback(titulo, callback)
    while True:
        vista = frame.copy()
        for indice, x in enumerate(posiciones_x):
            color = (255, 0, 0) if indice == 0 else (0, 0, 255)
            nombre = "A" if indice == 0 else "B"
            cv2.line(vista, (x, 0), (x, frame.shape[0] - 1), color, 2)
            cv2.putText(vista, nombre, (x + 7, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(vista, "Clic en A y luego B | Enter: confirmar | R/ESC: reiniciar",
                    (10, vista.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)
        cv2.imshow(titulo, vista)
        tecla = cv2.waitKey(20) & 0xFF
        if tecla in (ord("r"), 27):
            posiciones_x.clear()
        if len(posiciones_x) == 2 and tecla in (10, 13):
            break
    cv2.destroyWindow(titulo)
    return {
        "A": [(posiciones_x[0], 0), (posiciones_x[0], frame.shape[0] - 1)],
        "B": [(posiciones_x[1], 0), (posiciones_x[1], frame.shape[0] - 1)],
    }


def run_interactive_calibration(ruta_video):
    cap = cv2.VideoCapture(ruta_video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"No se pudo leer el video {ruta_video}")

    frame = redimensionar_estandar(frame)
    alto, ancho = frame.shape[:2]
    print("Seleccione las cuatro esquinas del rectángulo real en orden:")
    print("superior-izquierda, superior-derecha, inferior-derecha, inferior-izquierda.")
    puntos_imagen = _seleccionar(
        frame, 4, "Calibracion - 4 puntos",
        ["P1", "P2", "P3", "P4"],
    )
    ancho_m = float(input("Ancho real del rectángulo (m): ").replace(",", "."))
    alto_m = float(input("Largo real del rectángulo (m): ").replace(",", "."))
    if ancho_m <= 0 or alto_m <= 0:
        raise ValueError("Las dimensiones reales deben ser mayores que cero.")

    puntos_mundo = [(0, 0), (ancho_m, 0), (ancho_m, alto_m), (0, alto_m)]
    homografia = cv2.getPerspectiveTransform(
        np.asarray(puntos_imagen, dtype=np.float32),
        np.asarray(puntos_mundo, dtype=np.float32),
    )

    print("Seleccione con un clic la línea vertical A y luego la línea vertical B.")
    lineas = _seleccionar_lineas_verticales(frame)
    print("Seleccione dos esquinas opuestas de la ROI.")
    p_roi = _seleccionar(frame, 2, "Calibracion - ROI", ["R1", "R2"])
    x1, x2 = sorted((p_roi[0][0], p_roi[1][0]))
    y1, y2 = sorted((p_roi[0][1], p_roi[1][1]))

    resultado = {
        "date": datetime.now(timezone.utc).isoformat(),
        "video": os.path.basename(ruta_video),
        "resolution": [ancho, alto],
        "image_points": puntos_imagen,
        "world_points_m": puntos_mundo,
        "homography": homografia.tolist(),
        "line_segments": lineas,
        "roi": [x1, y1, x2 - x1, y2 - y1],
        "roi_norm": [x1 / ancho, y1 / alto, (x2 - x1) / ancho, (y2 - y1) / alto],
    }
    os.makedirs(CAL_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(ruta_video))[0]
    ruta = os.path.join(CAL_DIR, f"calibracion_{base}.json")
    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(resultado, archivo, indent=2)
    cv2.destroyAllWindows()
    print(f"Calibración guardada en: {ruta}")
    return ruta, True
