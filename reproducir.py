# -*- coding: utf-8 -*-
"""Reproduce desde el crudo el archivo entregado en la Competencia 1 y lo compara byte por byte.

Uso: `python reproducir.py`, con el crudo de la cátedra en `datos/competencia_01_crudo.csv` o en la ruta de la
variable de entorno COMPETENCIA_01_CRUDO.

Pasos:
1. controla el entorno y la huella SHA-256 del crudo;
2. corre las pruebas de los pasos 1 y 2;
3. `01_preparar_datos.py`: el dataset con las reparaciones de D01, en `datos/`;
4. `05_envios_ensamble.py entrenar d03k2`: 20 modelos LightGBM de clase K2 sobre la base C7, entrenados con
   202103–202106, que predicen 202108;
5. `05_envios_ensamble.py archivo d03k2 d04k2e 11500`: los 11.500 clientes con mayor promedio de probabilidades.

Cada etapa se compara con `huellas_esperadas.json`, tomadas de la corrida que generó el envío. Detienen la
reproducción sólo dos diferencias: un crudo distinto, porque ya no hay nada que comparar, y un CSV final distinto,
porque es lo que se entregó. Las diferencias intermedias se informan y se sigue: el parquet y las probabilidades
pueden cambiar con otra versión o en otro procesador sin que cambie la lista de clientes.

Salidas: las de cada paso, en `datos/` y `envios/`, y `envios/reproduccion.json` con las huellas y los tiempos.
"""
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
CRUDO = Path(os.environ.get("COMPETENCIA_01_CRUDO", AQUI / "datos" / "competencia_01_crudo.csv")).resolve()
ESPERADAS = json.loads((AQUI / "huellas_esperadas.json").read_text(encoding="utf-8"))
ENVIOS = AQUI / "envios"
N = 11_500
CSV = ENVIOS / "d04k2e" / f"d04k2en{N}.csv"
PYTHON_ORIGINAL = "3.13.3"


def huella(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def entorno():
    """Versiones instaladas de los paquetes de requirements.txt; si alguna difiere, se avisa y se sigue."""
    instaladas = {}
    for linea in (AQUI / "requirements.txt").read_text(encoding="utf-8").splitlines():
        linea = linea.split("#")[0].strip()
        if not linea:
            continue
        nombre, fijada = linea.split("==")
        try:
            instaladas[nombre] = importlib.metadata.version(nombre)
        except importlib.metadata.PackageNotFoundError:
            raise SystemExit(f"falta {nombre}: instalar con `pip install -r requirements.txt`")
        if instaladas[nombre] != fijada:
            print(f"AVISO: {nombre} {instaladas[nombre]}; el envío se generó con {fijada}")
    if platform.python_version() != PYTHON_ORIGINAL:
        print(f"AVISO: Python {platform.python_version()}; el envío se generó con {PYTHON_ORIGINAL}")
    return instaladas


def correr(pasos, *args):
    """Corre un paso con este mismo Python y el crudo ya controlado; si falla, la reproducción se detiene."""
    print(f"\n>>> python {' '.join(args)}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, *args], cwd=AQUI, env={**os.environ, "COMPETENCIA_01_CRUDO": str(CRUDO)})
    pasos.append({"paso": " ".join(args), "segundos": round(time.time() - t0, 1), "codigo": r.returncode})
    if r.returncode != 0:
        raise SystemExit(f"falló `python {' '.join(args)}` (código {r.returncode})")


def comparar(etapa, carpeta):
    """Compara las huellas de una etapa con las de la corrida original e informa las que difieren."""
    esperadas = ESPERADAS[etapa]
    distintas = [nombre for nombre, h in esperadas.items() if huella(carpeta / nombre) != h]
    if distintas:
        print(f"AVISO, {etapa}: {len(distintas)} de {len(esperadas)} archivos difieren de la corrida original "
              f"({', '.join(distintas[:3])}). Se sigue: lo que decide es el CSV.")
    elif len(esperadas) == 1:
        print(f"{etapa}: el archivo coincide con el de la corrida original")
    else:
        print(f"{etapa}: los {len(esperadas)} archivos coinciden con los de la corrida original")
    return {"coinciden": len(esperadas) - len(distintas), "total": len(esperadas), "distintos": distintas}


def main():
    t0 = time.time()
    resultado = {"inicio": time.strftime("%Y-%m-%d %H:%M:%S"), "sistema": platform.platform(),
                 "procesador": platform.processor(), "python": platform.python_version(), "paquetes": entorno()}
    if not CRUDO.exists():
        raise SystemExit(f"no encuentro el crudo en {CRUDO}: copiar competencia_01_crudo.csv a datos/ "
                         "o definir COMPETENCIA_01_CRUDO con su ruta")
    if huella(CRUDO) != ESPERADAS["crudo"]["competencia_01_crudo.csv"]:
        raise SystemExit(f"{CRUDO} no es el crudo de la cátedra: su SHA-256 no coincide")
    print("crudo: la huella coincide con la del dataset de la cátedra")
    for carpeta in ("d03k2", "d04k2e"):
        if (ENVIOS / carpeta).exists():
            raise SystemExit(f"envios/{carpeta} ya existe: para reproducir desde cero, borrarla o moverla antes")

    pasos = resultado["pasos"] = []
    correr(pasos, "test_01_preparar_datos.py")
    correr(pasos, "test_02_variables.py")
    correr(pasos, "01_preparar_datos.py")
    resultado["datos"] = comparar("datos", AQUI / "datos")
    correr(pasos, "05_envios_ensamble.py", "entrenar", "d03k2")
    resultado["modelos"] = comparar("modelos", ENVIOS / "d03k2")
    correr(pasos, "05_envios_ensamble.py", "archivo", "d03k2", "d04k2e", str(N))

    obtenida, esperada = huella(CSV), ESPERADAS["entrega"][CSV.name]
    resultado.update({"csv": CSV.name, "sha256": obtenida, "sha256_esperado": esperada,
                      "reproduce": obtenida == esperada, "minutos": round((time.time() - t0) / 60, 1)})
    (ENVIOS / "reproduccion.json").write_text(json.dumps(resultado, indent=1, ensure_ascii=False), encoding="utf-8")
    if obtenida != esperada:
        raise SystemExit(f"\nNO REPRODUCE: {CSV.name} tiene SHA-256 {obtenida}; se esperaba {esperada}")
    print(f"\nOK: {CSV.name} coincide byte por byte con el archivo entregado (SHA-256 {obtenida}), "
          f"en {resultado['minutos']} minutos")


if __name__ == "__main__":
    main()
