# eliminar_filas.py
import pandas as pd
from pathlib import Path


def eliminar_filas_con_texto(ruta_csv: str, texto: str):
    """
    Elimina todas las filas de 'ruta_csv' que contengan 'texto'
    en cualquiera de sus celdas y sobreescribe el archivo.

    Parámetros
    ----------
    ruta_csv : str
        Ruta al archivo CSV.
    texto : str
        Texto a buscar (comparación case-insensitive).
    """
    ruta = Path(ruta_csv)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_csv}")

    # Lee el CSV en un DataFrame (mantiene todo como texto)
    df = pd.read_csv(ruta, dtype=str, keep_default_na=False)

    # Crea una máscara booleana para cada fila:
    # True si al menos una celda contiene el texto
    texto_lower = texto.lower()
    mascara_filas_borrar = df.apply(
        lambda fila: fila.str.lower().str.contains(texto_lower).any(),
        axis=1
    )

    # Registra cuántas filas se eliminarán
    filas_iniciales = len(df)
    df_filtrado = df[~mascara_filas_borrar]
    filas_finales = len(df_filtrado)
    print(f"Filas iniciales: {filas_iniciales}")
    print(f"Filas eliminadas: {filas_iniciales - filas_finales}")
    print(f"Filas resultantes: {filas_finales}")

    # Sobrescribe el CSV
    df_filtrado.to_csv(ruta, index=False)
    print(f"✔ Archivo sobrescrito: {ruta_csv}")


# --------------------------------------------------------------------
# USO DE EJEMPLO
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Cambia estos valores a lo que necesites
    archivo = "Train_data.csv"
    texto_a_eliminar = "N/A"

    eliminar_filas_con_texto(archivo, texto_a_eliminar)
