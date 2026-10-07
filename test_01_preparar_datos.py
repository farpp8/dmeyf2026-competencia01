# -*- coding: utf-8 -*-
"""Pruebas de 01_preparar_datos.py con paneles chicos: clase, reparaciones, identificadores y rutas."""
import importlib.util
import shutil
import tempfile
from pathlib import Path

import duckdb

spec = importlib.util.spec_from_file_location("prep", Path(__file__).with_name("01_preparar_datos.py"))
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

MESES = [202103, 202104, 202105, 202106, 202107, 202108]
PANEL = {
    "siempre": MESES,
    "se_va_en_mayo": [202103, 202104],                           # BAJA+2 en marzo, BAJA+1 en abril
    "vuelve": [202103, 202105, 202106, 202107, 202108],          # falta en abril y vuelve: BAJA+1 en marzo
    "hueco_y_vuelve": [202103, 202104, 202106, 202107, 202108],  # BAJA+2 en marzo, BAJA+1 en abril
    "solo_julio": [202107],                                      # BAJA+1 en julio
    "solo_agosto": [202108],                                     # sin clase
}
ESPERADO = {
    ("siempre", 202103): "CONTINUA", ("siempre", 202106): "CONTINUA",
    ("siempre", 202107): None, ("siempre", 202108): None,
    ("se_va_en_mayo", 202103): "BAJA+2", ("se_va_en_mayo", 202104): "BAJA+1",
    ("vuelve", 202103): "BAJA+1", ("vuelve", 202105): "CONTINUA", ("vuelve", 202106): "CONTINUA",
    ("hueco_y_vuelve", 202103): "BAJA+2", ("hueco_y_vuelve", 202104): "BAJA+1",
    ("hueco_y_vuelve", 202106): "CONTINUA",
    ("solo_julio", 202107): "BAJA+1", ("solo_agosto", 202108): None,
}


def test_clase_como_la_catedra():
    con = duckdb.connect()
    filas = [(i, m) for i, (nombre, meses) in enumerate(PANEL.items()) for m in meses]
    con.execute("CREATE TABLE panel (numero_de_cliente BIGINT, foto_mes INTEGER)")
    con.executemany("INSERT INTO panel VALUES (?, ?)", filas)
    nombres = dict(enumerate(PANEL))
    obtenido = {(nombres[c], m): k for c, m, k in con.execute(prep.sql_clase("panel")).fetchall()}
    for clave, esperado in ESPERADO.items():
        assert obtenido[clave] == esperado, (clave, obtenido[clave], esperado)


def test_limites_de_las_reparaciones():
    """Financiación: NA sólo si supera estrictamente 10 veces un límite de compra positivo.
    Vencimiento: NA sólo por debajo de -1.000.000. Los NA de origen se mantienen."""
    con = duckdb.connect()
    casos = [  # (financiación, límite de compra, vencimiento, ¿financiación queda NA?, ¿vencimiento queda NA?)
        (1000.0, 100.0, -1_000_000, False, False),  # exactamente 10 veces y exactamente el centinela: no se tocan
        (1000.01, 100.0, -1_000_001, True, True),    # apenas por encima: se reparan
        (5000.0, 0.0, 10, False, False),            # límite de compra cero: no se aplica la regla
        (None, 100.0, None, True, True),            # NA de origen
        (50.0, None, -5, False, False),             # límite de compra NA: no se aplica la regla
    ]
    columnas = ["Visa_mfinanciacion_limite", "Visa_mlimitecompra", "Visa_Fvencimiento",
                "Master_mfinanciacion_limite", "Master_mlimitecompra", "Master_Fvencimiento"] + prep.SACAR
    con.execute("CREATE TABLE t (" + ", ".join(f"{c} DOUBLE" for c in columnas) + ")")
    for fin, compra, venc, _, _ in casos:
        con.execute("INSERT INTO t VALUES (" + ", ".join(["?"] * len(columnas)) + ")",
                    [fin, compra, venc, fin, compra, venc] + [1.0, 2.0] + [0.0] * (len(prep.SACAR) - 2))
    filas = con.execute(f"SELECT * FROM ({prep.sql_reparar('t')})").fetchall()
    nombres = [d[0] for d in con.description]
    for (fin, compra, venc, fin_na, venc_na), fila in zip(casos, filas):
        r = dict(zip(nombres, fila))
        for marca in ("Visa", "Master"):
            assert (r[f"{marca}_mfinanciacion_limite"] is None) == fin_na, (marca, fin, compra, r)
            assert (r[f"{marca}_Fvencimiento"] is None) == venc_na, (marca, venc, r)
        assert r["ccajas_depositos_otras"] == 3.0 and not set(prep.SACAR) & set(r), r


def test_identificador_en_notacion_cientifica(tmp="test_id.csv"):
    # Identificadores sintéticos, fuera del rango de los clientes del crudo
    path = Path(__file__).with_name(tmp)
    path.write_text("numero_de_cliente,foto_mes,x\n9.9e+07,202103,1\n99000001,202103,2\n", encoding="utf-8")
    try:
        con = duckdb.connect()
        filas = con.execute(f"SELECT numero_de_cliente, typeof(numero_de_cliente) FROM ({prep.sql_leer_crudo(path.as_posix())})").fetchall()
        assert filas == [(99000000, "BIGINT"), (99000001, "BIGINT")], filas
    finally:
        path.unlink()


def test_rutas_con_apostrofo():
    """Leer el crudo y escribir parquet en una carpeta con comilla simple en el nombre."""
    carpeta = Path(tempfile.mkdtemp(prefix="o'connor_"))
    try:
        crudo = carpeta / "crudo.csv"
        crudo.write_text("numero_de_cliente,foto_mes,x\n99000001,202103,1\n", encoding="utf-8")
        salida = carpeta / "salida.parquet"
        con = duckdb.connect()
        filas = con.execute(f"SELECT numero_de_cliente FROM ({prep.sql_leer_crudo(crudo.as_posix())})").fetchall()
        assert filas == [(99000001,)], filas
        con.execute(f"COPY (SELECT 1 AS x) TO {prep.literal(salida.as_posix())} (FORMAT parquet)")
        con.close()
        assert salida.exists()
    finally:
        shutil.rmtree(carpeta, ignore_errors=True)


if __name__ == "__main__":
    test_clase_como_la_catedra()
    test_limites_de_las_reparaciones()
    test_identificador_en_notacion_cientifica()
    test_rutas_con_apostrofo()
    print("OK: clase (14 casos), límites de reparación (5 casos), identificadores en notación científica y rutas "
          "con apóstrofo")
