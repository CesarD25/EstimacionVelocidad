import pandas as pd
from pathlib import Path


def guardar_csv(resultados, ruta_csv):
    Path(ruta_csv).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(resultados)
    df.to_csv(ruta_csv, index=False)
    return df


def guardar_excel(resultados, ruta_excel):
    Path(ruta_excel).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(resultados).to_excel(ruta_excel, index=False)
