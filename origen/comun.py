# -*- coding: utf-8 -*-
"""Lo que comparten los scripts de `origen/`: las rutas del repositorio, los módulos `02_variables.py`,
`03_envios_tablero.py` y `06_receta_denicolay.py`, la ganancia y la comparación con los resultados originales
(`origen/esperado/`)."""
import importlib.util
import os
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
ESPERADO = AQUI / "esperado"
SALIDAS = AQUI / "salidas"
DATOS = RAIZ / "datos" / "competencia_01_d01.parquet"
CRUDO = Path(os.environ.get("COMPETENCIA_01_CRUDO", RAIZ / "datos" / "competencia_01_crudo.csv"))


def modulo(nombre, ruta):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


var = modulo("variables", RAIZ / "02_variables.py")
tb = modulo("tablero", RAIZ / "03_envios_tablero.py")
receta_d = modulo("receta_d", RAIZ / "06_receta_denicolay.py")


def exigir_datos():
    if not DATOS.exists():
        raise SystemExit(f"falta {DATOS.relative_to(RAIZ)}: primero python 01_preparar_datos.py")


def curva(y, p):
    """G(k) en millones, k = 1..n: 1,1·TP(k) − 0,0275·k, con el ranking descendente de `p` y desempate estable."""
    orden = np.argsort(-np.asarray(p, dtype="float64"), kind="stable")
    return 1.1 * np.cumsum(np.asarray(y)[orden]) - 0.0275 * np.arange(1, len(y) + 1)


def a_rango(g):
    """A: la media de G(k) sobre los 7.001 enteros k = 8.000..15.000."""
    return float(np.mean(g[7999:15000]))


def comparar(nombre, obtenido, esperado, tolerancia):
    """Imprime la diferencia máxima entre dos listas de números y devuelve si está dentro de la tolerancia."""
    obtenido, esperado = np.asarray(obtenido, dtype="float64"), np.asarray(esperado, dtype="float64")
    dif = float(np.max(np.abs(obtenido - esperado))) if len(esperado) else 0.0
    ok = obtenido.shape == esperado.shape and dif <= tolerancia
    print(f"  {nombre}: diferencia máxima {dif:.3g} (tolerancia {tolerancia:g}) → {'coincide' if ok else 'NO COINCIDE'}")
    return ok
