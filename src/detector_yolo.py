"""Detección y seguimiento con un modelo YOLO preentrenado."""

from pathlib import Path


class DetectorYOLO:
    def __init__(self, modelo="yolov8n.pt", confianza=0.25, iou=0.45,
                 clases=None, dispositivo=None, tracker="bytetrack.yaml"):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Falta Ultralytics. Instale las dependencias con: "
                "pip install -r requirements.txt"
            ) from exc

        self.model = YOLO(str(Path(modelo)))
        self.confianza = float(confianza)
        self.iou = float(iou)
        self.clases = clases
        self.dispositivo = dispositivo
        self.tracker = tracker

    def detectar_y_seguir(self, frame):
        kwargs = {
            "source": frame,
            "persist": True,
            "tracker": self.tracker,
            "conf": self.confianza,
            "iou": self.iou,
            "verbose": False,
        }
        if self.clases is not None:
            kwargs["classes"] = self.clases
        if self.dispositivo not in (None, "", "auto"):
            kwargs["device"] = self.dispositivo

        resultado = self.model.track(**kwargs)[0]
        detecciones = []
        if resultado.boxes is None:
            return detecciones

        nombres = resultado.names
        ids = resultado.boxes.id
        for indice, caja in enumerate(resultado.boxes):
            x1, y1, x2, y2 = (int(v) for v in caja.xyxy[0].tolist())
            clase_id = int(caja.cls.item())
            track_id = int(ids[indice].item()) if ids is not None else None
            detecciones.append({
                "id": track_id,
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                # El punto de contacto con el suelo es más estable para homografía.
                "centro": ((x1 + x2) // 2, y2),
                "area": max(0, x2 - x1) * max(0, y2 - y1),
                "confianza": float(caja.conf.item()),
                "clase_id": clase_id,
                "clase": nombres.get(clase_id, str(clase_id)),
            })
        return detecciones
