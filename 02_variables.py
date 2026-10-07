# -*- coding: utf-8 -*-
"""Competencia 1, paso 2: variables según la decisión D02 (5 y 6-oct-2026).

Parte de `datos/competencia_01_d01.parquet` (paso 1). `construir()` arma el panel completo de variables; sin
opciones no transforma nada.

**Base vigente: `BASE`, es decir C7 = C0 + historia selectiva.** Usar `construir(df, **BASE)`. En la validación
local ningún bloque cumplió el umbral fijado antes, y el resultado fue C0. El 6-oct el usuario adoptó C7 por el
tablero público, que mide agosto: 84,77 contra 81,00 de C0, con 20 semillas cada uno.

La opción A fue aprobada primero y revertida después de la prueba, porque empeoró la validación de mayo en las
5 semillas. Sigue disponible como opción:
- `centrar=True`: `Visa_fultimo_cierre` y `Master_fultimo_cierre` pasan a la diferencia con la moda del mes,
  calculada entre los valores de 0 a 35 días; los NA se conservan;
- `mantenimiento=True`: `mcomisiones_mantenimiento` se reemplaza por dos indicadores, cobro (> 0) y devolución
  (< 0), que conservan los NA.

Copias exactas de versiones anteriores:
- la de las pruebas de D02 (valores por defecto = A): `experimentos/02_variables_usado_en_d02.py`;
- la de los envíos al tablero del 5-oct: `envios/02_variables_usado_en_envios_d02.py`.

- **Bloques optativos** (probados en D02, ninguno cumplió el umbral en la validación local):
  - `rangos`: ranking percentil dentro del mes de seis montos, conservando los originales;
  - `hist_selectiva`: *lag* 1 y delta 1 de seis conteos y del ranking mensual de cuatro montos (20 columnas);
  - `hist_amplia`: *lag* 1 y delta 1 de los predictores, con las exclusiones explícitas de `EXCLUIR_HIST_AMPLIA`;
  - `nuevas`: tres sumas Visa+Master, sus coberturas, saldo/límite por tarjeta y balance de transferencias;
  - `hist_1y2_todas`: *lag* y delta de orden 1 y 2 de todos los predictores, la receta de Denicolay (Zulip
    #192079). Se usa en el envío 3 al tablero.

La historia une por cliente y mes exacto (mes calendario anterior): marzo queda en NA y nunca se completa con
el presente ni con meses futuros. Los rankings usan sólo los valores de su propio mes.
Decisión: `decisiones/D02_variables/decision.md`.
"""
import numpy as np
import pandas as pd

CLAVES = ["numero_de_cliente", "foto_mes"]
CLASE = "clase_ternaria"
CIERRES = ["Visa_fultimo_cierre", "Master_fultimo_cierre"]
DIAS = CIERRES + ["Visa_Fvencimiento", "Master_Fvencimiento", "Visa_fechaalta", "Master_fechaalta",
                  "Visa_Finiciomora", "Master_Finiciomora"]
MANTENIMIENTO = "mcomisiones_mantenimiento"
RANGOS = ["mpayroll", "mpayroll2", "mcomisiones", "mcomisiones_otras", "mrentabilidad", "mrentabilidad_annual"]
HIST_CONTEOS = ["ctrx_quarter", "cproductos", "ctarjeta_debito_transacciones", "ctarjeta_visa_transacciones",
                "ctarjeta_master_transacciones", "chomebanking_transacciones"]
HIST_MONTOS = ["mcuentas_saldo", "mpayroll", "mtransferencias_recibidas", "mtransferencias_emitidas"]
# Historia amplia: exclusiones exactas de la especificación conjunta (prueba_conjunta.json). Quedan fuera los
# estados y la mora (cierre transitorio de julio), el VIP (pico de junio), las variables en días y los descuentos
# (pico de julio). Edad y antigüedad se incluyen: la revisión de Codex marcó que excluirlas se apartaba de la
# especificación.
EXCLUIR_HIST_AMPLIA = {"Visa_status", "Master_status", "Visa_delinquency", "Master_delinquency", "cliente_vip",
                       "mtarjeta_visa_descuentos", "mtarjeta_master_descuentos", *DIAS}
SUMAS = {"suma_msaldototal": ("Visa_msaldototal", "Master_msaldototal"),
         "suma_mlimitecompra": ("Visa_mlimitecompra", "Master_mlimitecompra"),
         "suma_cconsumos": ("Visa_cconsumos", "Master_cconsumos")}
# Base vigente desde el 6-oct: C7 (decisión del usuario sobre el tablero; decisiones/D02_variables/decision.md).
BASE = {"hist_selectiva": True}


def _periodo(foto_mes):
    return (foto_mes // 100) * 12 + foto_mes % 100


def centrar_cierres(df):
    """Días desde el último cierre menos la moda del mes, calculada entre los valores de 0 a 35."""
    for v in CIERRES:
        validos = df.loc[df[v].between(0, 35), ["foto_mes", v]]
        moda = validos.groupby("foto_mes")[v].agg(lambda s: s.value_counts().idxmax())
        df[v] = df[v] - df["foto_mes"].map(moda)
    return df


def indicadores_mantenimiento(df):
    m = df.pop(MANTENIMIENTO)
    df["mantenimiento_cobro"] = np.where(m.isna(), np.nan, (m > 0).astype(float))
    df["mantenimiento_devolucion"] = np.where(m.isna(), np.nan, (m < 0).astype(float))
    return df


def rango_mes(df, v):
    return df.groupby("foto_mes")[v].rank(pct=True)


def historia(df, columnas, ordenes=(1,)):
    """lagk = valor del mismo cliente k meses calendario antes; deltak = actual − lagk. Sin ese mes, NA.

    Con `ordenes=(1, 2)`, el *lag* 2 existe aunque falte el mes intermedio: es el mes exacto t−2.
    """
    # Claves en int64 de los dos lados: con foto_mes INTEGER (int32) del parquet, pandas no puede unir.
    per = _periodo(df["foto_mes"].astype("int64")).to_numpy()
    cliente = df["numero_de_cliente"].astype("int64").to_numpy()
    actual = pd.DataFrame({"numero_de_cliente": cliente, "_per": per})
    salida = {}
    for k in ordenes:
        previo = df[columnas].copy()
        previo["numero_de_cliente"] = cliente
        previo["_per"] = per + k
        lag = actual.merge(previo, on=["numero_de_cliente", "_per"], how="left", validate="one_to_one")
        for v in columnas:
            lagk = lag[v].to_numpy(dtype="float64")
            salida[f"{v}_lag{k}"] = lagk.astype("float32")
            salida[f"{v}_delta{k}"] = (df[v].to_numpy(dtype="float64") - lagk).astype("float32")
    return pd.DataFrame(salida, index=df.index)


def nuevas_variables(df):
    out = {}
    for nombre, (a, b) in SUMAS.items():
        out[nombre] = df[[a, b]].sum(axis=1, min_count=1)
        out[nombre.replace("suma_", "cobertura_")] = df[[a, b]].notna().sum(axis=1).astype(float)
    for t in ("Visa", "Master"):
        lim = df[f"{t}_mlimitecompra"]
        lim = lim.where((lim > 0) & np.isfinite(lim))          # denominador positivo y finito; si no, NA
        out[f"{t}_saldo_limite"] = df[f"{t}_msaldototal"] / lim
    rec, emi = df["mtransferencias_recibidas"], df["mtransferencias_emitidas"]
    den = rec.abs() + emi.abs()
    den = den.where((den > 0) & np.isfinite(den))
    out["balance_transferencias"] = (rec - emi) / den
    return pd.DataFrame(out, index=df.index)


def construir(df, centrar=False, mantenimiento=False, rangos=False, hist_selectiva=False, hist_amplia=False,
              nuevas=False, hist_1y2_todas=False):
    """Devuelve el panel con claves, clase y predictores. No modifica `df`."""
    faltan = {*CIERRES, MANTENIMIENTO, *RANGOS, *HIST_CONTEOS, *HIST_MONTOS, *(c for p in SUMAS.values() for c in p)} - set(df.columns)
    if faltan:
        raise KeyError(f"faltan columnas: {sorted(faltan)}")
    df = df.sort_values(CLAVES).reset_index(drop=True).copy()
    if centrar:
        df = centrar_cierres(df)
    if mantenimiento:
        df = indicadores_mantenimiento(df)
    predictores = [c for c in df.columns if c not in CLAVES + [CLASE]]
    bloques = [df]
    if rangos:
        bloques.append(pd.DataFrame({f"{v}_rango": rango_mes(df, v) for v in RANGOS}))
    if hist_selectiva:
        r = pd.DataFrame({f"{v}_rango": rango_mes(df, v) for v in HIST_MONTOS})
        base = pd.concat([df[CLAVES + HIST_CONTEOS], r], axis=1)
        bloques.append(historia(base, HIST_CONTEOS + list(r.columns)))
    if hist_amplia:
        bloques.append(historia(df, [c for c in predictores if c not in EXCLUIR_HIST_AMPLIA]))
    if nuevas:
        bloques.append(nuevas_variables(df))
    if hist_1y2_todas:
        bloques.append(historia(df, predictores, ordenes=(1, 2)))
    return pd.concat(bloques, axis=1)


def predictores_de(panel):
    return [c for c in panel.columns if c not in CLAVES + [CLASE]]
