import cv2
import numpy as np

# Variable global para guardar las coordenadas X en el orden EXACTO de tus clics
coordenadas_lineas = []

def marcar_lineas(event, x, y, flags, param):
    """
    Función que escucha los clics del mouse para dibujar las líneas A y B.
    """
    global coordenadas_lineas
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(coordenadas_lineas) < 2:
            # Guardamos la coordenada X exactamente en el orden en que haces clic
            coordenadas_lineas.append(x) 
            
            # 1er clic = Azul (A) | 2do clic = Rojo (B)
            color = (255, 0, 0) if len(coordenadas_lineas) == 1 else (0, 0, 255)
            etiqueta = "A" if len(coordenadas_lineas) == 1 else "B"
            
            # Dibujamos la línea
            cv2.line(param, (x, 0), (x, param.shape[0]), color, 2)
            
            # Etiqueta simple y un poco más grande (escala 1 en lugar de 0.6)
            cv2.putText(param, etiqueta, (x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow("Modulo 3: Calibracion de Zona", param)

def main():
    # =================================================================
    # MÓDULO 1: Captura de video
    # =================================================================
    nombre_video = "pelotita_prueba.mp4" 
    cap = cv2.VideoCapture(nombre_video)

    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video '{nombre_video}'.")
        return

    ret, frame_fondo = cap.read()
    if not ret:
        print("Error al leer el primer fotograma.")
        return

    # =================================================================
    # MÓDULO 3: Definición de zona de medición (Puntos A y B)
    # =================================================================
    print("\n" + "="*50)
    print(" MÓDULO 3: DEFINICIÓN DE ZONA DE MEDICIÓN")
    print("="*50)
    print(">>> Haz CLIC en la ventana de video para marcar las zonas:")
    print("    - 1er Clic: Línea A (Punto de Inicio)")
    print("    - 2do Clic: Línea B (Punto Final)")

    frame_calibracion = frame_fondo.copy()
    cv2.imshow("Modulo 3: Calibracion de Zona", frame_calibracion)
    cv2.setMouseCallback("Modulo 3: Calibracion de Zona", marcar_lineas, frame_calibracion)

    # Esperamos a que hagas los 2 clics
    while len(coordenadas_lineas) < 2:
        cv2.waitKey(10)
        
    cv2.destroyWindow("Modulo 3: Calibracion de Zona")

    # ASIGNACIÓN DIRECTA: Respetamos tu orden de clics
    linea_a_x = coordenadas_lineas[0]
    linea_b_x = coordenadas_lineas[1]

    # Solicitar distancia física explícitamente en MILÍMETROS
    distancia_str = input("\n>>> Ingresa la distancia física entre A y B (en milímetros, ej: 300): ")
    try:
        distancia_real_mm = float(distancia_str)
    except ValueError:
        print("Error: Debes ingresar un número. Usando 300 mm por defecto.")
        distancia_real_mm = 300.0

    print(f"\n[ZONA CONFIGURADA] A: X={linea_a_x} | B: X={linea_b_x} | Distancia: {distancia_real_mm} mm")
    print(">>> Reproduciendo video a 1/4 de velocidad... Presiona 'q' para salir.\n")

    # =================================================================
    # MÓDULO 2: Preprocesamiento de la imagen (Fondo)
    # =================================================================
    fondo_gris = cv2.cvtColor(frame_fondo, cv2.COLOR_BGR2GRAY)
    fondo_gris = cv2.GaussianBlur(fondo_gris, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)

    while True:
        ret, frame_actual = cap.read()
        if not ret: break

        frame_gris = cv2.cvtColor(frame_actual, cv2.COLOR_BGR2GRAY)
        frame_gris = cv2.GaussianBlur(frame_gris, (5, 5), 0)

        # =================================================================
        # MÓDULO 4: Detección del objeto
        # =================================================================
        diferencia = cv2.absdiff(fondo_gris, frame_gris)
        _, binarizada = cv2.threshold(diferencia, 25, 255, cv2.THRESH_BINARY)

        mascara_limpia = cv2.morphologyEx(binarizada, cv2.MORPH_OPEN, kernel)
        mascara_final = cv2.morphologyEx(mascara_limpia, cv2.MORPH_CLOSE, kernel)

        contornos, _ = cv2.findContours(mascara_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_dibujado = frame_actual.copy()
        
        # Redibujamos las etiquetas "A" y "B" actualizadas en el bucle
        cv2.line(frame_dibujado, (linea_a_x, 0), (linea_a_x, frame_dibujado.shape[0]), (255, 0, 0), 2)
        cv2.putText(frame_dibujado, "A", (linea_a_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.line(frame_dibujado, (linea_b_x, 0), (linea_b_x, frame_dibujado.shape[0]), (0, 0, 255), 2)
        cv2.putText(frame_dibujado, "B", (linea_b_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        pelotita_detectada = False
        
        for c in contornos:
            if cv2.contourArea(c) > 200: 
                pelotita_detectada = True
                x, y, w, h = cv2.boundingRect(c)
                
                cv2.rectangle(frame_dibujado, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cx = int(x + w / 2)
                cy = int(y + h / 2)
                cv2.circle(frame_dibujado, (cx, cy), 4, (0, 255, 0), -1)
                
                texto_coordenada = f"X: {cx}"
                cv2.putText(frame_dibujado, texto_coordenada, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # =================================================================
        # SALIDA VISUAL INTEGRADA
        # =================================================================
        mascara_bgr = cv2.cvtColor(mascara_final, cv2.COLOR_GRAY2BGR)
        pantalla_dividida = np.hstack((frame_dibujado, mascara_bgr))

        cv2.imshow("Sistema de Deteccion Integrado", pantalla_dividida)

        if cv2.waitKey(120) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()