"""Estimación de velocidad con YOLO preentrenado y ByteTrack."""

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.calibracion_interactiva import run_interactive_calibration
from src.captura import (
    abrir_video,
    leer_propiedades,
    redimensionar_estandar,
    resolucion_estandar,
)
from src.configuracion import cargar_configuracion
from src.cruces import detectar_cruces
from src.detector_yolo import DetectorYOLO
from src.evaluacion import calcular_error
from src.exportacion import guardar_csv, guardar_excel
from src.lineas import dibujar_lineas, transformar_punto
from src.roi import dibujar_roi
from src.velocidad import calcular_velocidad
from src.visualizacion import dibujar_deteccion


def _ruta_calibracion(video):
    if video:
        base = Path(video).stem
        candidata = Path("calibraciones") / f"calibracion_{base}.json"
        if candidata.exists():
            return candidata
    return None


def _cargar_calibracion(ruta):
    if not ruta:
        raise FileNotFoundError(
            "No hay calibración para este video. Ejecute primero: "
            "python main.py y seleccione la opción de calibración."
        )
    with open(ruta, encoding="utf-8") as archivo:
        datos = json.load(archivo)
    if not datos.get("line_segments"):
        raise ValueError("La calibración no contiene las líneas A/B.")
    return datos


def _ajustar_calibracion(calibracion, resolucion_objetivo):
    """Escala calibraciones antiguas a la resolución estándar de trabajo."""
    anterior = calibracion.get("resolution")
    if not anterior or tuple(anterior) == tuple(resolucion_objetivo):
        return calibracion
    ancho_anterior, alto_anterior = anterior
    ancho_nuevo, alto_nuevo = resolucion_objetivo
    sx, sy = ancho_nuevo / ancho_anterior, alto_nuevo / alto_anterior
    datos = dict(calibracion)

    def escalar_punto(punto):
        return [punto[0] * sx, punto[1] * sy]

    datos["line_segments"] = {
        nombre: [escalar_punto(p) for p in calibracion["line_segments"][nombre]]
        for nombre in ("A", "B")
    }
    if calibracion.get("image_points"):
        datos["image_points"] = [
            escalar_punto(p) for p in calibracion["image_points"]
        ]
    if calibracion.get("roi"):
        x, y, w, h = calibracion["roi"]
        datos["roi"] = [x * sx, y * sy, w * sx, h * sy]
    if calibracion.get("homography"):
        h = np.asarray(calibracion["homography"], dtype=float)
        escala_inversa = np.array([
            [1 / sx, 0, 0],
            [0, 1 / sy, 0],
            [0, 0, 1],
        ])
        datos["homography"] = (h @ escala_inversa).tolist()
    datos["resolution"] = [ancho_nuevo, alto_nuevo]
    return datos


def _distancia_lineas(calibracion):
    homografia = calibracion.get("homography")
    lineas = calibracion["line_segments"]
    if homografia:
        h = np.asarray(homografia, dtype=float)
        centros = []
        for nombre in ("A", "B"):
            p1, p2 = lineas[nombre]
            centros.append(((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2))
        a_m = transformar_punto(centros[0], h)
        b_m = transformar_punto(centros[1], h)
        return float(np.linalg.norm(np.asarray(a_m) - np.asarray(b_m)))
    escala = calibracion.get("scale_m_per_px")
    if escala:
        a, b = lineas["A"], lineas["B"]
        ca = np.mean(np.asarray(a), axis=0)
        cb = np.mean(np.asarray(b), axis=0)
        return float(np.linalg.norm(ca - cb) * float(escala))
    raise ValueError("La calibración requiere homografía o scale_m_per_px.")


def _en_roi(punto, roi):
    x, y, w, h = roi
    return x <= punto[0] <= x + w and y <= punto[1] <= y + h


def _velocidad_referencia(video):
    if not video:
        return None
    ruta = Path(video)
    metadata = ruta.parent / "metadata" / f"{ruta.stem}.yaml"
    if not metadata.exists():
        return None
    with open(metadata, encoding="utf-8") as archivo:
        valor = yaml.safe_load(archivo).get("Velocidad_de_referencia_m_s")
    try:
        return float(valor) if valor not in ("", None) else None
    except (TypeError, ValueError):
        return None


def procesar(config, video=None, mostrar=None):
    fuente = video
    cap = abrir_video(fuente)
    props = leer_propiedades(cap)
    resolucion = resolucion_estandar(props["ancho"], props["alto"])
    ruta_cal = _ruta_calibracion(fuente)
    calibracion = _ajustar_calibracion(
        _cargar_calibracion(ruta_cal), resolucion
    )
    lineas = {
        nombre: [
            (int(round(p[0])), int(round(p[1])))
            for p in calibracion["line_segments"][nombre]
        ]
        for nombre in ("A", "B")
    }
    roi = tuple(int(v) for v in calibracion.get(
        "roi", [0, 0, resolucion[0], resolucion[1]]
    ))
    distancia_m = _distancia_lineas(calibracion)
    velocidad_referencia = _velocidad_referencia(fuente)

    det = config["deteccion"]
    if det.get("metodo", "yolo").lower() != "yolo":
        raise ValueError("Esta versión usa exclusivamente el detector YOLO.")
    detector = DetectorYOLO(
        modelo=det.get("modelo", "yolov8n.pt"),
        confianza=det.get("confianza", 0.25),
        iou=det.get("iou", 0.45),
        clases=det.get("clases"),
        dispositivo=det.get("dispositivo"),
        tracker=det.get("tracker", "bytetrack.yaml"),
    )
    movimiento = det.get("movimiento", {})
    ventana_movimiento = int(movimiento.get("ventana_frames", 6))
    minimo_frames = int(movimiento.get("minimo_frames", 3))
    desplazamiento_minimo = float(
        movimiento.get("desplazamiento_minimo_px", 12)
    )

    salida = config["salida"]
    nombre = Path(fuente).stem
    ruta_video = Path(salida.get("directorio_videos", "videos_salida")) / f"{nombre}_resultado.mp4"
    ruta_csv = Path(salida.get("directorio_resultados", "resultados")) / f"{nombre}_mediciones.csv"
    ruta_excel = Path(salida.get("directorio_resultados", "resultados")) / f"{nombre}_mediciones.xlsx"
    ruta_video.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    if config.get("visualizacion", {}).get("guardar_video", True):
        writer = cv2.VideoWriter(
            str(ruta_video), cv2.VideoWriter_fourcc(*"mp4v"), props["fps"],
            resolucion,
        )
        if not writer.isOpened():
            raise IOError(f"No se pudo crear el video de salida: {ruta_video}")

    mostrar = config.get("visualizacion", {}).get("mostrar", True) if mostrar is None else mostrar
    estados, mediciones = {}, []
    siguiente_id_visible = 1
    fotograma = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fotograma += 1
        frame = redimensionar_estandar(frame)
        detecciones_yolo = [
            d for d in detector.detectar_y_seguir(frame)
            if d["id"] is not None and _en_roi(d["centro"], roi)
        ]
        detecciones = []
        for d in detecciones_yolo:
            estado = estados.setdefault(d["id"], {
                "anterior": None,
                "cruces": {},
                "clase": d["clase"],
                "historial": deque(maxlen=ventana_movimiento),
            })
            estado["historial"].append(d["centro"])
            historial = estado["historial"]
            desplazamiento = (
                float(np.linalg.norm(
                    np.asarray(historial[-1]) - np.asarray(historial[0])
                ))
                if len(historial) >= minimo_frames else 0.0
            )
            en_movimiento = desplazamiento >= desplazamiento_minimo
            if en_movimiento:
                if "id_visible" not in estado:
                    estado["id_visible"] = siguiente_id_visible
                    siguiente_id_visible += 1
                deteccion_visible = dict(d)
                deteccion_visible["_track_id"] = d["id"]
                deteccion_visible["id"] = estado["id_visible"]
                detecciones.append(deteccion_visible)
            if en_movimiento and estado["anterior"] is not None:
                for nombre, cruzo in detectar_cruces(
                    estado["anterior"], d["centro"], lineas["A"], lineas["B"]
                ).items():
                    if cruzo and nombre not in estado["cruces"]:
                        estado["cruces"][nombre] = fotograma
            estado["anterior"] = d["centro"]

            if en_movimiento and {"A", "B"} <= estado["cruces"].keys() and not estado.get("medido"):
                inicio = min(estado["cruces"].values())
                fin = max(estado["cruces"].values())
                metrica = calcular_velocidad(distancia_m, props["fps"], inicio, fin)
                fila = {
                    "track_id": estado["id_visible"], "clase": estado["clase"],
                    "direccion": "A->B" if estado["cruces"]["A"] < estado["cruces"]["B"] else "B->A",
                    "frame_a": estado["cruces"]["A"], "frame_b": estado["cruces"]["B"],
                    "distancia_m": distancia_m, **metrica,
                    "velocidad_km_h": metrica["velocidad_m_s"] * 3.6,
                }
                if velocidad_referencia is not None:
                    fila.update(calcular_error(
                        metrica["velocidad_m_s"], velocidad_referencia
                    ))
                mediciones.append(fila)
                estado["medido"] = fila

        dibujar_lineas(frame, *lineas["A"], *lineas["B"])
        dibujar_roi(frame, roi)
        for d in detecciones:
            dibujar_deteccion(frame, d)
            medida = estados[d["_track_id"]].get("medido")
            if medida:
                x, y = d["centro"]
                cv2.putText(frame, f"{medida['velocidad_km_h']:.1f} km/h",
                            (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 255, 255), 2)
        if writer:
            writer.write(frame)
        if mostrar:
            cv2.imshow("YOLO + ByteTrack - Estimacion de velocidad", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    guardar_csv(mediciones, ruta_csv)
    guardar_excel(mediciones, ruta_excel)
    print(f"Procesados {fotograma} fotogramas; mediciones: {len(mediciones)}")
    print(f"CSV: {ruta_csv} | Excel: {ruta_excel} | Video: {ruta_video}")
    return mediciones


def _listar_videos():
    extensiones = {".mp4", ".avi", ".mov", ".mkv"}
    carpeta = Path("videos_entrada")
    return sorted(
        ruta for ruta in carpeta.iterdir()
        if ruta.is_file() and ruta.suffix.lower() in extensiones
    ) if carpeta.exists() else []


def _elegir_video():
    videos = _listar_videos()
    print("\nVideos disponibles:")
    for indice, video in enumerate(videos, 1):
        print(f"  {indice}) {video.name}")
    print("  0) Escribir otra ruta")
    while True:
        opcion = input("Seleccione el video: ").strip()
        if opcion == "0":
            ruta = Path(input("Ruta del video: ").strip().strip('"'))
            if ruta.is_file():
                return str(ruta)
            print("El archivo no existe.")
        elif opcion.isdigit() and 1 <= int(opcion) <= len(videos):
            return str(videos[int(opcion) - 1])
        else:
            print("Opción inválida.")


def _elegir_modo():
    print("Sistema de estimación de velocidad")
    print("  1) Calibrar un video")
    print("  2) Ejecutar un video")
    while True:
        opcion = input("Seleccione una opción [1/2]: ").strip()
        if opcion in {"1", "2"}:
            return opcion
        print("Opción inválida.")


def main():
    config = cargar_configuracion()
    modo = _elegir_modo()
    video = _elegir_video()
    if modo == "1":
        run_interactive_calibration(video)
    else:
        procesar(config, video)


if __name__ == "__main__":
    main()
