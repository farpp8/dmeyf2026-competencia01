# -*- coding: utf-8 -*-
"""Pruebas de 02_variables.py con paneles chicos."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("var", Path(__file__).with_name("02_variables.py"))
var = importlib.util.module_from_spec(spec)
spec.loader.exec_module(var)


def panel_minimo(filas):
    """Panel con todas las columnas que exige construir(); las no indicadas valen 1."""
    columnas = {*var.CIERRES, var.MANTENIMIENTO, *var.RANGOS, *var.HIST_CONTEOS, *var.HIST_MONTOS,
                *(c for p in var.SUMAS.values() for c in p), "Visa_status", "cliente_edad"}
    base = {c: 1.0 for c in columnas}
    df = pd.DataFrame([{**base, **f} for f in filas])
    df["clase_ternaria"] = None
    return df


def test_cierres_centrados_con_moda_del_mes():
    filas = [{"numero_de_cliente": i, "foto_mes": 202103, "Visa_fultimo_cierre": v, "Master_fultimo_cierre": np.nan}
             for i, v in enumerate([1, 1, 1, 7, 21, 99])]
    out = var.centrar_cierres(panel_minimo(filas))
    assert out.Visa_fultimo_cierre.tolist() == [0, 0, 0, 6, 20, 98]  # moda 1 entre 0 y 35; el 99 no cuenta para la moda
    assert out.Master_fultimo_cierre.isna().all()


def test_indicadores_de_mantenimiento_conservan_na():
    filas = [{"numero_de_cliente": i, "foto_mes": 202103, var.MANTENIMIENTO: v} for i, v in enumerate([5.0, -3.0, 0.0, np.nan])]
    out = var.indicadores_mantenimiento(panel_minimo(filas))
    assert var.MANTENIMIENTO not in out
    assert out.mantenimiento_cobro.tolist()[:3] == [1.0, 0.0, 0.0] and np.isnan(out.mantenimiento_cobro.iloc[3])
    assert out.mantenimiento_devolucion.tolist()[:3] == [0.0, 1.0, 0.0] and np.isnan(out.mantenimiento_devolucion.iloc[3])


def test_historia_usa_el_mes_calendario_exacto():
    # Cliente 1: marzo, abril, junio (falta mayo). Cliente 2: sólo abril.
    filas = [{"numero_de_cliente": 1, "foto_mes": 202103, "ctrx_quarter": 10.0},
             {"numero_de_cliente": 1, "foto_mes": 202104, "ctrx_quarter": 7.0},
             {"numero_de_cliente": 1, "foto_mes": 202106, "ctrx_quarter": 3.0},
             {"numero_de_cliente": 2, "foto_mes": 202104, "ctrx_quarter": 5.0}]
    df = panel_minimo(filas).sort_values(var.CLAVES).reset_index(drop=True)
    h = var.historia(df, ["ctrx_quarter"])
    got = dict(zip(zip(df.numero_de_cliente, df.foto_mes), zip(h.ctrx_quarter_lag1, h.ctrx_quarter_delta1)))
    assert np.isnan(got[(1, 202103)][0])                 # marzo: sin historia
    assert got[(1, 202104)] == (10.0, -3.0)              # abril usa marzo
    assert np.isnan(got[(1, 202106)][0])                 # junio no salta a abril: mayo falta
    assert np.isnan(got[(2, 202104)][0])


def test_historia_de_orden_2_usa_el_mes_exacto():
    # Presente en marzo y mayo, falta abril: en mayo el lag 1 es NA y el lag 2 es el valor de marzo.
    filas = [{"numero_de_cliente": 1, "foto_mes": 202103, "ctrx_quarter": 10.0},
             {"numero_de_cliente": 1, "foto_mes": 202105, "ctrx_quarter": 4.0}]
    df = panel_minimo(filas).sort_values(var.CLAVES).reset_index(drop=True)
    h = var.historia(df, ["ctrx_quarter"], ordenes=(1, 2))
    mayo = h.iloc[1]
    assert np.isnan(mayo.ctrx_quarter_lag1) and mayo.ctrx_quarter_lag2 == 10.0 and mayo.ctrx_quarter_delta2 == -6.0
    assert np.isnan(h.iloc[0].ctrx_quarter_lag2)


def test_por_defecto_no_transforma_nada():
    """Decisión D02: C0. Sin opciones, construir() deja cierres y mantenimiento como vienen de D01."""
    filas = [{"numero_de_cliente": i, "foto_mes": 202103, "Visa_fultimo_cierre": 7.0, var.MANTENIMIENTO: -5.0}
             for i in range(3)]
    p = var.construir(panel_minimo(filas))
    assert p.Visa_fultimo_cierre.tolist() == [7.0] * 3 and p[var.MANTENIMIENTO].tolist() == [-5.0] * 3
    assert "mantenimiento_cobro" not in p and not any(c.endswith(("_lag1", "_rango")) for c in p)


def test_base_vigente_es_c7():
    """Decisión del 6-oct: la base es C0 + historia selectiva, 20 columnas de lag 1 y delta 1."""
    assert var.BASE == {"hist_selectiva": True}
    filas = [{"numero_de_cliente": 1, "foto_mes": m, "ctrx_quarter": v} for m, v in [(202103, 10.0), (202104, 7.0)]]
    p = var.construir(panel_minimo(filas), **var.BASE)
    historia = [c for c in p if c.endswith(("_lag1", "_delta1"))]
    assert len(historia) == 20 and "mcuentas_saldo_rango_lag1" in historia
    assert p.loc[p.foto_mes == 202104, "ctrx_quarter_delta1"].item() == -3.0
    assert p.Visa_fultimo_cierre.tolist() == [1.0, 1.0]          # sin cierres centrados


def test_historia_con_foto_mes_int32():
    """El parquet guarda foto_mes como INTEGER (int32): la unión tiene que funcionar igual."""
    filas = [{"numero_de_cliente": 1, "foto_mes": 202103, "ctrx_quarter": 10.0},
             {"numero_de_cliente": 1, "foto_mes": 202104, "ctrx_quarter": 7.0}]
    df = panel_minimo(filas)
    df["foto_mes"] = df["foto_mes"].astype("int32")
    df["numero_de_cliente"] = df["numero_de_cliente"].astype("int64")
    h = var.historia(df.sort_values(var.CLAVES).reset_index(drop=True), ["ctrx_quarter"])
    assert np.isnan(h.ctrx_quarter_lag1.iloc[0]) and h.ctrx_quarter_lag1.iloc[1] == 10.0


def test_nuevas_variables_y_sus_na():
    filas = [{"numero_de_cliente": 1, "foto_mes": 202103, "Visa_msaldototal": 50.0, "Master_msaldototal": np.nan,
              "Visa_mlimitecompra": 100.0, "Master_mlimitecompra": 0.0,
              "mtransferencias_recibidas": 30.0, "mtransferencias_emitidas": 10.0},
             {"numero_de_cliente": 2, "foto_mes": 202103, "Visa_msaldototal": np.nan, "Master_msaldototal": np.nan,
              "mtransferencias_recibidas": 0.0, "mtransferencias_emitidas": 0.0}]
    filas.append({"numero_de_cliente": 3, "foto_mes": 202103, "Visa_msaldototal": 10.0, "Master_msaldototal": 10.0,
                  "Visa_mlimitecompra": -100.0, "Master_mlimitecompra": np.inf,
                  "mtransferencias_recibidas": np.inf, "mtransferencias_emitidas": 1.0})
    n = var.nuevas_variables(panel_minimo(filas))
    assert n.suma_msaldototal.iloc[0] == 50.0 and n.cobertura_msaldototal.iloc[0] == 1.0  # suma parcial
    assert np.isnan(n.suma_msaldototal.iloc[1]) and n.cobertura_msaldototal.iloc[1] == 0.0
    assert n.Visa_saldo_limite.iloc[0] == 0.5 and np.isnan(n.Master_saldo_limite.iloc[0])  # límite 0: NA
    assert n.balance_transferencias.iloc[0] == 0.5 and np.isnan(n.balance_transferencias.iloc[1])
    assert np.isnan(n.Visa_saldo_limite.iloc[2]) and np.isnan(n.Master_saldo_limite.iloc[2])  # negativo o infinito: NA
    assert np.isnan(n.balance_transferencias.iloc[2])


def test_rango_usa_solo_su_mes():
    filas = [{"numero_de_cliente": i, "foto_mes": m, "mpayroll": v}
             for i, (m, v) in enumerate([(202103, 10.0), (202103, 20.0), (202104, 1000.0), (202104, 2000.0)])]
    r = var.rango_mes(panel_minimo(filas), "mpayroll")
    assert r.tolist() == [0.5, 1.0, 0.5, 1.0]


if __name__ == "__main__":
    for nombre, f in list(globals().items()):
        if nombre.startswith("test_"):
            f()
    print("OK: cierres, mantenimiento, historia por mes exacto, variables nuevas y rangos por mes")
