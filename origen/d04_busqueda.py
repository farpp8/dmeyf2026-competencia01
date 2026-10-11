# -*- coding: utf-8 -*-
"""Origen de los hiperparámetros de D, paso 1: la búsqueda bayesiana de la D04 (6 y 7-oct-2026).

Es el script que corrió la búsqueda (`experimentos/d04_busqueda/correr_busqueda.py` de la carpeta de trabajo), con
estos cambios, que no alteran los resultados:
- las rutas: lee `02_variables.py` y `datos/competencia_01_d01.parquet` de este repositorio, y escribe en
  `origen/salidas/`;
- no tiene los límites de tiempo del original (8 horas o el 8-oct a las 06:00). La corrida original terminó las 40
  pruebas antes de llegar a ellos, así que no cambian nada. El de fecha, hoy, cortaría la búsqueda en el acto;
- `--hasta K` corta después de K pruebas completas, en una carpeta propia, para comprobar una parte;
- **no reanuda.** El muestreador de Optuna no guarda el estado de su generador en el estudio: si la búsqueda se corta
  y se reanuda, vuelve a sortear desde la semilla y toma otro camino. La corrida original fue de corrido, así que el
  script exige una carpeta vacía;
- al final compara cada prueba con la corrida original (`origen/esperado/pruebas_d04.csv`).

**Qué busca:**
- **Fijos:** la receta de Denicolay (`learning_rate` 0,005, `max_bin` 31, `min_data_in_leaf` 0,
  `feature_fraction` 0,5, `feature_fraction_bynode` 0,2), sin *bagging*, 12 hilos y `deterministic`.
- **Libres:** `num_leaves` (16–255) y el hessiano de referencia h (0,5–128), en escala logarítmica.
  - `min_sum_hessian_in_leaf` = h · filas de entrenamiento / 326.184.
  - 326.184 son las filas de entrenamiento de F2 (202103–202104). En F2, h es exactamente el mínimo de hessiano por
    hoja; en los demás cortes se escala por sus filas.
- **Datos:** la base C7 (`construir(df, **BASE)`), con la clase K2, en dos cortes:
  - F1: entrena con 202103 y valida en 202105;
  - F2: entrena con 202103–202104 y valida en 202106.
- **Rondas:** cada ajuste llega a 8.000, y se evalúan los prefijos de 1.000, 2.000, 3.000, 5.000 y 8.000.
- **Valor de cada prueba:** el máximo sobre las rondas de A = (A_F1 + A_F2)/2.
  - A_f es la media de G(k) = 1.100.000·TP(k) − 27.500·k sobre los enteros de 8.000 a 15.000.
  - TP cuenta sólo los BAJA+2, con el ranking del promedio de las dos semillas (147347 y 322607).
- **Muestreo:** TPE con semilla 147347: 3 anclas (31, 63 y 255 hojas con h = 8), 9 al azar y 28 adaptativas, hasta
  40 pruebas completas.

Uso, desde la raíz del repositorio y después de `python 01_preparar_datos.py`:
    python origen/d04_busqueda.py            # las 40 pruebas: unas 3 horas (la original tardó 3,0 h)
    python origen/d04_busqueda.py --hasta 4  # las primeras 4 (las 3 anclas y la primera al azar): unos 17 minutos
    python origen/d04_busqueda.py --humo     # prueba de humo en una carpeta temporal, con pocas rondas

Si se corta, hay que borrar su carpeta de salida y empezar de nuevo.

Salidas en `origen/salidas/d04_busqueda/` (o en `d04_busqueda_hasta<K>/`, con `--hasta K`):
- `pruebas.csv`, una fila por prueba;
- `predicciones/`, las probabilidades de cada semilla y corte con las rondas elegidas;
- `manifiesto.json` y el estudio, en `busqueda.db`.
"""
import datetime as dt
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

import comun

var = comun.var
DATOS = comun.DATOS
SALIDA = comun.SALIDAS / "d04_busqueda"
ESPERADO = comun.ESPERADO / "pruebas_d04.csv"

FIJOS = dict(objective="binary", learning_rate=0.005, max_bin=31, min_data_in_leaf=0, feature_fraction=0.5,
             feature_fraction_bynode=0.2, bagging_fraction=1.0, bagging_freq=0, feature_pre_filter=False,
             verbosity=-1, num_threads=12, deterministic=True, force_col_wise=True)
FILAS_REF = 326_184
CORTES = {"F1": ([202103], 202105), "F2": ([202103, 202104], 202106)}
SEMILLAS = [147347, 322607]
RONDAS = [1000, 2000, 3000, 5000, 8000]
POSITIVAS = ["BAJA+1", "BAJA+2"]
HOJAS = (16, 255)
H_REF = (0.5, 128.0)
ANCLAS = [{"num_leaves": 31, "h_ref": 8.0}, {"num_leaves": 63, "h_ref": 8.0}, {"num_leaves": 255, "h_ref": 8.0}]
N_PRUEBAS, N_INICIALES, SEMILLA_TPE = 40, 12, 147347
GANANCIA_TP, COSTO_ENVIO = 1_100_000, 27_500


def curva_ganancia(y, p):
    """G(k) en pesos enteros, k = 1..n, ordenando por probabilidad descendente (empates en orden estable)."""
    orden = np.argsort(-np.asarray(p, dtype="float64"), kind="stable")
    return GANANCIA_TP * np.cumsum(np.asarray(y, dtype="int64")[orden]) - COSTO_ENVIO * np.arange(1, len(y) + 1)


def a_rango(y, p):
    """A en millones: media de G(k) sobre los 7.001 enteros k = 8.000..15.000."""
    return float(curva_ganancia(y, p)[7999:15000].mean()) / 1e6


def auc(y, p):
    y = np.asarray(y)
    r = pd.Series(np.asarray(p, dtype="float64")).rank(method="average").to_numpy()
    npos = y.sum()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))


def prefijos(modelo, X, rondas):
    """Probabilidades con los primeros T árboles, para cada T de `rondas`, sumando los puntajes crudos por bloques."""
    crudo, previo, salida = np.zeros(X.shape[0]), 0, {}
    for t in rondas:
        crudo = crudo + modelo.predict(X, start_iteration=previo, num_iteration=t - previo, raw_score=True)
        previo = t
        salida[t] = 1.0 / (1.0 + np.exp(-crudo))
    return salida


def huella(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preparar():
    """Matrices de entrenamiento y validación de cada corte, con las columnas útiles de su entrenamiento."""
    comun.exigir_datos()
    panel = var.construir(pd.read_parquet(DATOS), **var.BASE)
    panel = panel[panel.foto_mes.between(202103, 202106)].reset_index(drop=True)
    predictores = var.predictores_de(panel)
    X = panel[predictores].astype("float32")
    clase = panel[var.CLASE].to_numpy(dtype="U8")
    if set(np.unique(clase)) != {"BAJA+1", "BAJA+2", "CONTINUA"}:
        raise SystemExit("clases inesperadas en 202103–202106")
    datos = {}
    for corte, (meses, mes_val) in CORTES.items():
        en_train, en_val = panel.foto_mes.isin(meses).to_numpy(), (panel.foto_mes == mes_val).to_numpy()
        cols = [c for c in predictores if X.loc[en_train, c].nunique(dropna=False) >= 2]
        y_train = np.isin(clase[en_train], POSITIVAS).astype("int8")
        ds = lgb.Dataset(np.ascontiguousarray(X.loc[en_train, cols].to_numpy()), label=y_train,
                         params={"max_bin": FIJOS["max_bin"], "feature_pre_filter": False, "verbosity": -1},
                         free_raw_data=False)
        ds.construct()
        datos[corte] = {"ds": ds, "X_val": np.ascontiguousarray(X.loc[en_val, cols].to_numpy()),
                        "y_val": (clase[en_val] == "BAJA+2").astype("int8"), "clase_val": clase[en_val],
                        "claves_val": panel.loc[en_val, "numero_de_cliente"].to_numpy(dtype="int64"),
                        "filas": int(en_train.sum()), "columnas": cols}
    return datos


def evaluar(datos, hojas, h_ref, rondas, salida=None, numero=None):
    """Entrena las semillas en los dos cortes y devuelve A por ronda, la ronda elegida y los detalles."""
    probs = {c: {s: None for s in SEMILLAS} for c in CORTES}
    segundos = 0.0
    for corte, d in datos.items():
        params = {**FIJOS, "num_leaves": int(hojas), "min_sum_hessian_in_leaf": h_ref * d["filas"] / FILAS_REF}
        for s in SEMILLAS:
            t0 = time.time()
            modelo = lgb.train({**params, "seed": s}, d["ds"], num_boost_round=rondas[-1])
            probs[corte][s] = prefijos(modelo, d["X_val"], rondas)
            segundos += time.time() - t0
    a_por_ronda = {}
    for t in rondas:
        a_f = {c: a_rango(datos[c]["y_val"], np.mean([probs[c][s][t] for s in SEMILLAS], axis=0)) for c in CORTES}
        a_por_ronda[t] = (a_f, float(np.mean(list(a_f.values()))))
    mejor_t = max(rondas, key=lambda t: (a_por_ronda[t][1], -t))       # ante un empate exacto, menos rondas
    detalle = {"rondas": mejor_t, "A": a_por_ronda[mejor_t][1], "segundos": round(segundos, 1)}
    for c in CORTES:
        detalle[f"A_{c}"] = a_por_ronda[mejor_t][0][c]
        detalle[f"auc_{c}"] = auc(datos[c]["y_val"], np.mean([probs[c][s][mejor_t] for s in SEMILLAS], axis=0))
        for s in SEMILLAS:
            detalle[f"A_{c}_{s}"] = a_rango(datos[c]["y_val"], probs[c][s][mejor_t])
    for t in rondas:
        detalle[f"A_r{t}"] = a_por_ronda[t][1]
    if salida is not None:
        for c in CORTES:
            for s in SEMILLAS:
                destino = salida / f"p{numero:03d}_{c}_{s}.npy"
                temporal = destino.with_suffix(".tmp.npy")
                np.save(temporal, probs[c][s][mejor_t].astype("float64"), allow_pickle=False)
                temporal.replace(destino)
    return detalle


def manifiesto_actual(rondas):
    config = {"fijos": FIJOS, "filas_ref": FILAS_REF, "cortes": CORTES, "semillas": SEMILLAS, "rondas": rondas,
              "positivas": POSITIVAS, "hojas": HOJAS, "h_ref": H_REF, "anclas": ANCLAS, "n_pruebas": N_PRUEBAS,
              "n_iniciales": N_INICIALES, "semilla_tpe": SEMILLA_TPE, "base": var.BASE}
    return {"datos_sha256": huella(DATOS), "variables_py_sha256": huella(comun.RAIZ / "02_variables.py"),
            "script_sha256": huella(__file__),
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "lightgbm": lgb.__version__, "optuna": optuna.__version__, "numpy": np.__version__}


def correr(carpeta, rondas, n_pruebas, nueva=True):
    """`nueva`: exige una carpeta sin estudio, porque reanudar cambia el sorteo de Optuna."""
    if nueva and ((carpeta / "busqueda.db").exists() or (carpeta / "pruebas.csv").exists()):
        raise SystemExit(f"{carpeta} ya tiene una búsqueda. Reanudarla no reproduce la original, porque el muestreador "
                         "vuelve a sortear desde la semilla: borrar la carpeta y correr de nuevo, de corrido")
    carpeta.mkdir(parents=True, exist_ok=True)
    pred = carpeta / "predicciones"
    pred.mkdir(exist_ok=True)
    manifiesto = carpeta / "manifiesto.json"
    actual = manifiesto_actual(rondas)
    if manifiesto.exists():
        previo = json.loads(manifiesto.read_text(encoding="utf-8"))
        if any(previo.get(k) != v for k, v in actual.items()):
            raise SystemExit("la búsqueda cambió desde que empezó: usar una carpeta nueva")
    else:
        manifiesto.write_text(json.dumps(actual, indent=1), encoding="utf-8")
    t0 = time.time()
    datos = preparar()
    for c, d in datos.items():
        for nombre in ("y_val", "clase_val", "claves_val"):
            destino = pred / f"{nombre.split('_')[0]}_{c}.npy"
            if destino.exists():
                if not np.array_equal(np.load(destino, allow_pickle=False), d[nombre]):
                    raise SystemExit(f"{destino.name} no coincide")
            else:
                np.save(destino, d[nombre], allow_pickle=False)
    print(f"datos listos en {time.time() - t0:.0f} s: " +
          ", ".join(f"{c} {d['filas']} filas y {len(d['columnas'])} columnas" for c, d in datos.items()), flush=True)
    sampler = optuna.samplers.TPESampler(seed=SEMILLA_TPE, n_startup_trials=N_INICIALES)
    estudio = optuna.create_study(study_name="d04", storage=f"sqlite:///{(carpeta / 'busqueda.db').as_posix()}",
                                  direction="maximize", sampler=sampler, load_if_exists=True)
    if len(estudio.trials) == 0:
        for ancla in ANCLAS:
            estudio.enqueue_trial(ancla)

    def objetivo(trial):
        hojas = trial.suggest_int("num_leaves", HOJAS[0], HOJAS[1], log=True)
        h_ref = trial.suggest_float("h_ref", H_REF[0], H_REF[1], log=True)
        detalle = evaluar(datos, hojas, h_ref, rondas, salida=pred, numero=trial.number)
        for k, v in detalle.items():
            trial.set_user_attr(k, v)
        fila = {"prueba": trial.number, "num_leaves": hojas, "h_ref": h_ref, **detalle,
                "fin": dt.datetime.now().isoformat(timespec="seconds")}
        archivo = carpeta / "pruebas.csv"
        pd.DataFrame([fila]).to_csv(archivo, mode="a", header=not archivo.exists(), index=False)
        print(f"prueba {trial.number}: hojas {hojas}, h {h_ref:.3g}, rondas {detalle['rondas']}, A {detalle['A']:.2f} M "
              f"(F1 {detalle['A_F1']:.2f}, F2 {detalle['A_F2']:.2f}), {detalle['segundos']:.0f} s", flush=True)
        return detalle["A"]

    while True:
        completas = [t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completas) >= n_pruebas:
            print(f"fin: {len(completas)} pruebas completas", flush=True)
            break
        estudio.optimize(objetivo, n_trials=1, catch=())
    mejor = estudio.best_trial
    print(f"mejor hasta acá: prueba {mejor.number}, {mejor.params}, rondas {mejor.user_attrs['rondas']}, "
          f"A {mejor.value:.2f} M", flush=True)


def comparar_con_original(carpeta):
    """Cada prueba corrida contra la misma prueba de la corrida original: las hojas, h, las rondas y A."""
    obtenidas = pd.read_csv(carpeta / "pruebas.csv").drop_duplicates("prueba", keep="last").set_index("prueba")
    originales = pd.read_csv(ESPERADO).set_index("prueba").loc[obtenidas.index]
    print(f"\nComparación con la corrida original ({len(obtenidas)} pruebas):")
    ok = (obtenidas.num_leaves == originales.num_leaves).all() and (obtenidas.rondas == originales.rondas).all()
    print(f"  hojas y rondas: {'iguales' if ok else 'DISTINTAS'}")
    ok &= comun.comparar("h", obtenidas.h_ref, originales.h_ref, 1e-12)
    for col in ("A", "A_F1", "A_F2"):
        ok &= comun.comparar(col, obtenidas[col], originales[col], 1e-9)
    print("→ " + ("reproduce la búsqueda original" if ok else "NO reproduce la búsqueda original"))
    return ok


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if "--humo" in sys.argv[1:]:
        d = Path(tempfile.mkdtemp())          # en Windows, SQLite mantiene abierto el estudio hasta salir
        correr(d, rondas=[10, 20], n_pruebas=4)
        print(pd.read_csv(d / "pruebas.csv")[["prueba", "num_leaves", "h_ref", "rondas", "A", "A_r10", "A_r20"]])
    else:
        hasta = int(sys.argv[sys.argv.index("--hasta") + 1]) if "--hasta" in sys.argv else N_PRUEBAS
        carpeta = SALIDA if hasta >= N_PRUEBAS else comun.SALIDAS / f"d04_busqueda_hasta{hasta}"
        correr(carpeta, RONDAS, min(hasta, N_PRUEBAS))
        if not comparar_con_original(carpeta):
            raise SystemExit(1)
