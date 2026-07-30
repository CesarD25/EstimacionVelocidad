def lado_linea(punto, linea):
    (x1, y1), (x2, y2) = linea
    return (x2 - x1) * (punto[1] - y1) - (y2 - y1) * (punto[0] - x1)


def segmento_cruza_linea(anterior, actual, linea):
    lado_anterior = lado_linea(anterior, linea)
    lado_actual = lado_linea(actual, linea)
    return (lado_anterior < 0 <= lado_actual) or (lado_anterior > 0 >= lado_actual)


def detectar_cruces(anterior, actual, linea_a, linea_b):
    return {
        "A": segmento_cruza_linea(anterior, actual, linea_a),
        "B": segmento_cruza_linea(anterior, actual, linea_b),
    }
