"""Apertura y normalización de videos."""

import os

import cv2


def abrir_video(fuente):
    if not os.path.exists(fuente):
        raise FileNotFoundError(f"No existe el video: {fuente}")
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {fuente}")
    return cap


def leer_propiedades(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((codec_int >> 8 * i) & 0xFF) for i in range(4))
    if fps <= 0:
        raise ValueError("El FPS debe ser mayor que cero.")
    if total_frames <= 0:
        raise ValueError("El video no contiene fotogramas.")
    return {
        "fps": fps,
        "ancho": ancho,
        "alto": alto,
        "total_frames": total_frames,
        "duracion": total_frames / fps,
        "codec": codec,
    }


def resolucion_estandar(ancho, alto):
    """Devuelve 16:9 horizontal o 9:16 vertical."""
    return (540, 960) if alto > ancho else (960, 540)


def redimensionar_estandar(frame):
    ancho_objetivo, alto_objetivo = resolucion_estandar(
        frame.shape[1], frame.shape[0]
    )
    interpolacion = (
        cv2.INTER_AREA
        if frame.shape[1] > ancho_objetivo or frame.shape[0] > alto_objetivo
        else cv2.INTER_LINEAR
    )
    return cv2.resize(
        frame, (ancho_objetivo, alto_objetivo), interpolation=interpolacion
    )
