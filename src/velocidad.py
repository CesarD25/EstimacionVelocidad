def calcular_velocidad(distancia_metros, fps, fotograma_inicio, fotograma_fin):
    if fps <= 0:
        raise ValueError("FPS debe ser mayor que cero.")
    tiempo_segundos = (fotograma_fin - fotograma_inicio) / fps
    velocidad_m_s = distancia_metros / tiempo_segundos if tiempo_segundos > 0 else 0
    return {'tiempo_s': tiempo_segundos, 'velocidad_m_s': velocidad_m_s}
