"""Métricas reproducibles para comparar velocidad estimada y real."""


def calcular_error(estimada_m_s, referencia_m_s):
    estimada = float(estimada_m_s)
    referencia = float(referencia_m_s)
    absoluto = abs(estimada - referencia)
    porcentual = (absoluto / abs(referencia) * 100) if referencia != 0 else None
    return {
        "velocidad_referencia_m_s": referencia,
        "error_absoluto_m_s": absoluto,
        "error_porcentual": porcentual,
    }
