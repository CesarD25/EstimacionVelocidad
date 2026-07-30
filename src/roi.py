import cv2


def dibujar_roi(frame, roi):
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.putText(frame, 'ROI', (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
