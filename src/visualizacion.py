import cv2


def dibujar_deteccion(frame, deteccion, color=(0, 255, 0)):
    x, y, w, h = deteccion['bbox']
    cx, cy = deteccion['centro']
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.circle(frame, (cx, cy), 4, color, -1)
    etiqueta = f"ID {deteccion.get('id', '?')} {deteccion.get('clase', '')}"
    if 'confianza' in deteccion:
        etiqueta += f" {deteccion['confianza']:.2f}"
    cv2.putText(frame, etiqueta, (x, max(18, y - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
