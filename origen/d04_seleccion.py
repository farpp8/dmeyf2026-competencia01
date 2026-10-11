# -*- coding: utf-8 -*-
"""Origen de los hiperparámetros de D, paso 2: la selección de la D04 con 5 semillas (7-oct-2026).

Es la etapa `seleccion` de `experimentos/d04_busqueda/correr_etapas.py` de la carpeta de trabajo, adaptada a este
repositorio. De acá sale la prueba 38: 83 hojas y h = 12,791817301585011.

**Qué hace:**
1. **Las finalistas:** las 5 pruebas de mayor A de la búsqueda, sin repetir hojas y h. En un empate, menos rondas ×
   hojas, después menos rondas y después el número de prueba.
   - Las toma de `origen/salidas/d04_busqueda/pruebas.csv` si la búsqueda está completa (40 pruebas).
   - Si no, de la corrida original, `origen/esperado/pruebas_d04.csv`.
2. **Cada finalista, con las 5 semillas del usuario,** en F1 y F2, con sus rondas (las 5 eligieron 1.000).
   - Con finalistas de la búsqueda propia, para las semillas 147347 y 322607 usa sus predicciones, como el original.
   - Con las de la corrida original, entrena las 5: las predicciones de una búsqueda propia podrían ser de otras
     pruebas.
3. **La referencia, «la K2 actual»:** la configuración vigente de la D03, con las mismas 5 semillas y en los mismos
   cortes.
   - `learning_rate` 0,05, 63 hojas, `min_data_in_leaf` 200, `feature_fraction` 0,5 y *bagging* 0,8 con
     frecuencia 1;
   - 500 rondas;
   - un `Dataset` por semilla, sin parámetros propios, como en `experimentos/d03_clase/correr.py`.
4. **La comparación:** ΔA del ensamble por corte, la diferencia media por semilla y cuántas semillas mejoran.
   - La regla de la D04 exigía ΔA ≥ 10 M, mejorar F1 y F2, y 4 de 5 semillas en cada corte.
   - **Ninguna finalista la cumplió.** La prueba 38 fue la de mayor ΔA (+6,19 M) y la que se llevó a D en la D06.
5. **El control:** compara la tabla con la original (`origen/esperado/seleccion_d04.csv`).

Uso, desde la raíz y después de `python 01_preparar_datos.py`: `python origen/d04_seleccion.py`. Sin la búsqueda
completa, entrena 50 modelos de las finalistas y 10 de la referencia: unos 10 minutos (la original, que reutilizó 20
predicciones de la búsqueda, tardó 5).

Salidas en `origen/salidas/d04_seleccion/`: `seleccion.csv` y las predicciones.
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import comun
import d04_busqueda as bq

SALIDA = comun.SALIDAS / "d04_seleccion"
PRED_BUSQUEDA = bq.SALIDA / "predicciones"
ESPERADO = comun.ESPERADO / "seleccion_d04.csv"
SEMILLAS_5 = [147347, 322607, 462467, 782149, 983809]
N_FINALISTAS, MARGEN_A = 5, 10.0
# la «K2 actual» de la D03 (experimentos/d03_clase/correr.py)
PARAMS_D03 = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.5,
                  bagging_fraction=0.8, bagging_freq=1, verbosity=-1, num_threads=12, deterministic=True,
                  force_col_wise=True)
RONDAS_D03 = 500


def validar(p, n, nombre):
    p = np.asarray(p)
    if p.shape != (n,) or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise SystemExit(f"{nombre}: predicción inválida")
    return p


def pruebas():
    """Las pruebas de la búsqueda completa corrida acá («propia»), o si no, las de la corrida original."""
    propia = bq.SALIDA / "pruebas.csv"
    if propia.exists():
        p = pd.read_csv(propia).drop_duplicates("prueba", keep="last")
        if len(p) >= bq.N_PRUEBAS:
            print(f"finalistas de la búsqueda corrida acá ({propia.relative_to(comun.RAIZ)})", flush=True)
            return p, "propia"
    print(f"finalistas de la corrida original ({ESPERADO.parent.relative_to(comun.RAIZ)}/pruebas_d04.csv)", flush=True)
    return pd.read_csv(bq.ESPERADO), "original"


def finalistas():
    todas, origen = pruebas()
    p = todas[["prueba", "num_leaves", "h_ref", "rondas", "A"]].copy()
    p["costo"] = p.rondas * p.num_leaves
    p = p.sort_values(["A", "costo", "rondas", "prueba"], ascending=[False, True, True, True])
    return p.drop_duplicates(["num_leaves", "h_ref"]).head(N_FINALISTAS), origen


def guardar(path, p):
    temporal = path.with_suffix(".tmp.npy")
    np.save(temporal, p.astype("float64"), allow_pickle=False)
    temporal.replace(path)


def prediccion_finalista(datos, fila, corte, s, origen):
    """Con finalistas de la búsqueda propia, usa sus predicciones (semillas 147347 y 322607), como el original. Con
    las de la corrida original, entrena: las predicciones de una búsqueda propia podrían ser de otras pruebas."""
    d = datos[corte]
    carpetas = ([PRED_BUSQUEDA] if origen == "propia" else []) + [SALIDA / origen]
    for carpeta in carpetas:
        path = carpeta / f"p{int(fila.prueba):03d}_{corte}_{s}.npy"
        if path.exists():
            return validar(np.load(path, allow_pickle=False), len(d["y_val"]), path.name)
    t0 = time.time()
    params = {**bq.FIJOS, "num_leaves": int(fila.num_leaves),
              "min_sum_hessian_in_leaf": fila.h_ref * d["filas"] / bq.FILAS_REF, "seed": s}
    p = validar(lgb.train(params, d["ds"], num_boost_round=int(fila.rondas)).predict(d["X_val"]), len(d["y_val"]),
                f"prueba {int(fila.prueba)} {corte} {s}")
    (SALIDA / origen).mkdir(parents=True, exist_ok=True)
    guardar(SALIDA / origen / f"p{int(fila.prueba):03d}_{corte}_{s}.npy", p)
    print(f"  prueba {int(fila.prueba)} {corte} {s}: A {bq.a_rango(d['y_val'], p):.2f} M, {time.time() - t0:.0f} s",
          flush=True)
    return p


def prediccion_referencia(datos, corte, s):
    """La K2 actual de la D03: un `Dataset` nuevo por semilla, sin parámetros propios, y los de la D03 al entrenar."""
    d = datos[corte]
    path = SALIDA / f"K2_{corte}_{s}.npy"
    if path.exists():
        return validar(np.load(path, allow_pickle=False), len(d["y_val"]), path.name)
    t0 = time.time()
    ds = lgb.Dataset(d["X_train"], label=d["y_train"], free_raw_data=False)
    p = validar(lgb.train({**PARAMS_D03, "seed": s}, ds, num_boost_round=RONDAS_D03).predict(d["X_val"]),
                len(d["y_val"]), f"K2 {corte} {s}")
    guardar(path, p)
    print(f"  K2 actual {corte} {s}: A {bq.a_rango(d['y_val'], p):.2f} M, {time.time() - t0:.0f} s", flush=True)
    return p


def preparar():
    """Los datos de la búsqueda (`bq.preparar()`), más las matrices de entrenamiento que necesita la referencia."""
    datos = bq.preparar()
    for d in datos.values():
        d["X_train"] = d["ds"].get_data()
        d["y_train"] = d["ds"].get_label().astype("int8")
    return datos


def seleccion():
    SALIDA.mkdir(parents=True, exist_ok=True)
    fins, origen = finalistas()
    print(f"finalistas: {fins.prueba.tolist()}", flush=True)
    datos = preparar()
    filas = []
    for fila in fins.itertuples(index=False):
        comp = {}
        for corte in bq.CORTES:
            y = datos[corte]["y_val"]
            nueva = {s: prediccion_finalista(datos, fila, corte, s, origen) for s in SEMILLAS_5}
            actual = {s: prediccion_referencia(datos, corte, s) for s in SEMILLAS_5}
            a_n = bq.a_rango(y, np.mean(list(nueva.values()), axis=0))
            a_a = bq.a_rango(y, np.mean(list(actual.values()), axis=0))
            ind = [bq.a_rango(y, nueva[s]) - bq.a_rango(y, actual[s]) for s in SEMILLAS_5]
            comp[corte] = {"A_nueva": a_n, "A_actual": a_a, "dif_ensamble": a_n - a_a,
                           "dif_individual_media": float(np.mean(ind)), "semillas_mejoran": int(sum(x > 0 for x in ind)),
                           "por_semilla": " ".join(f"{x:+.1f}" for x in ind)}
        d_a = float(np.mean([comp[c]["dif_ensamble"] for c in bq.CORTES]))
        ok = (d_a >= MARGEN_A and all(comp[c]["dif_ensamble"] > 0 for c in bq.CORTES)
              and all(comp[c]["dif_individual_media"] > 0 and comp[c]["semillas_mejoran"] >= 4 for c in bq.CORTES))
        for c in bq.CORTES:
            filas.append({"prueba": int(fila.prueba), "hojas": int(fila.num_leaves), "h_ref": round(fila.h_ref, 3),
                          "rondas": int(fila.rondas), "corte": c, "A_nueva_M": round(comp[c]["A_nueva"], 2),
                          "A_actual_M": round(comp[c]["A_actual"], 2), "dif_ensamble_M": round(comp[c]["dif_ensamble"], 2),
                          "dif_individual_media_M": round(comp[c]["dif_individual_media"], 2),
                          "semillas_que_mejoran": f"{comp[c]['semillas_mejoran']}/5", "por_semilla_M": comp[c]["por_semilla"],
                          "dA_M": round(d_a, 2), "elegible": "sí" if ok else "no"})
    t = pd.DataFrame(filas)
    t.to_csv(SALIDA / "seleccion.csv", index=False)
    print(t[["prueba", "hojas", "h_ref", "corte", "A_nueva_M", "A_actual_M", "dif_ensamble_M", "semillas_que_mejoran",
             "dA_M", "elegible"]].to_string(index=False))
    mejor = t.drop_duplicates("prueba").sort_values("dA_M", ascending=False).iloc[0]
    print(f"\nMayor ΔA: prueba {mejor.prueba} ({mejor.hojas} hojas, h {mejor.h_ref}), ΔA {mejor.dA_M:+.2f} M; "
          f"elegibles: {', '.join(map(str, t[t.elegible == 'sí'].prueba.unique())) or 'ninguna'}")
    return t


def comparar_con_original(t):
    o = pd.read_csv(ESPERADO)
    print("\nComparación con la selección original:")
    ok = t[["prueba", "corte", "hojas", "rondas", "semillas_que_mejoran", "elegible"]].equals(
        o[["prueba", "corte", "hojas", "rondas", "semillas_que_mejoran", "elegible"]])
    print(f"  finalistas, cortes, semillas que mejoran y elegibilidad: {'iguales' if ok else 'DISTINTAS'}")
    for col in ("A_nueva_M", "A_actual_M", "dif_ensamble_M", "dA_M"):
        ok &= comun.comparar(col, t[col], o[col], 0.011)        # la tabla original está redondeada a 2 decimales
    print("→ " + ("reproduce la selección original" if ok else "NO reproduce la selección original"))
    return ok


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if not comparar_con_original(seleccion()):
        raise SystemExit(1)
