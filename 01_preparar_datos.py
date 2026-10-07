# -*- coding: utf-8 -*-
"""Competencia 1, paso 1: dataset reparado según la decisión D01 (opción A, 5-oct-2026).

Parte del dataset original de la cátedra (`competencia_01_crudo.csv`), para que la entrega se pueda
reproducir desde cero. Aplica:

1. `clase_ternaria` con la lógica de la cátedra (notebook z470): falta en el mes siguiente = BAJA+1, aunque
   después vuelva; está en el siguiente pero no en el subsiguiente = BAJA+2. 202107 sólo admite BAJA+1 y
   202108 queda sin clase.
2. `ccajas_depositos + ccajas_otras` → `ccajas_depositos_otras`, y se sacan las componentes. Entre abril y
   junio las transacciones por caja se reclasificaron entre las dos.
3. Se sacan `Visa_mconsumototal` y `Master_mconsumototal`, idénticos a `*_mconsumospesos`.
4. NA en `*_mfinanciacion_limite` cuando `*_mlimitecompra > 0` y la financiación supera 10 veces el límite de
   compra: errores de escala de un mes por cliente.
5. NA en `*_Fvencimiento` menores que −1.000.000: valor centinela.
6. Se sacan `cprestamos_personales` y `mprestamos_personales`: en junio cambia su relación con la baja.

No se modifican los NA de descuentos, `cliente_vip` ni los estados de tarjeta.

El crudo escribe un identificador en notación científica, del tipo `9.9e+07`, en seis filas. Por eso
`numero_de_cliente` se lee como texto y se convierte a entero exacto (BIGINT); si alguna conversión no es exacta,
el proceso se detiene.

Salidas, en `datos/`:
- `competencia_01_d01.parquet`: los seis meses, con las reparaciones y la clase;
- `reparaciones_d01.parquet`: el valor original de cada celda modificada;
- `resumen_d01.json`: cantidades por regla y mes, clase por mes, versiones y huella del crudo.

Decisión y análisis: `decisiones/D01_datos/`. Revisión de código de Codex: `decisiones/D01_datos/revision_codigo_codex.md`.
"""
import hashlib
import json
import os
import platform
from pathlib import Path

import duckdb

AQUI = Path(__file__).resolve().parent
# El crudo de la cátedra no se incluye en el repositorio: va en datos/, o en la ruta de COMPETENCIA_01_CRUDO.
CRUDO = Path(os.environ.get("COMPETENCIA_01_CRUDO", AQUI / "datos" / "competencia_01_crudo.csv"))
SALIDA = AQUI / "datos"

MESES = [202103, 202104, 202105, 202106, 202107, 202108]
UMBRAL_FINANCIACION = 10
CENTINELA_VENCIMIENTO = -1_000_000
SACAR = ["ccajas_depositos", "ccajas_otras", "Visa_mconsumototal", "Master_mconsumototal",
         "cprestamos_personales", "mprestamos_personales"]
REPARADAS = ["Visa_mfinanciacion_limite", "Master_mfinanciacion_limite", "Visa_Fvencimiento", "Master_Fvencimiento"]

# Verificado en D01 contra la definición literal: el archivo viejo marcaba CONTINUA a 29 clientes que faltan
# un mes y vuelven (`decisiones/D01_datos/cruce_claude.md`).
ESPERADO_BAJA2 = {202103: 960, 202104: 1139, 202105: 870, 202106: 1098}
ESPERADO_BAJA1 = {202103: 1019, 202104: 964, 202105: 1143, 202106: 874, 202107: 1103}


def exigir(condicion, mensaje):
    """Control que no se apaga con `python -O`, a diferencia de assert."""
    if not condicion:
        raise RuntimeError(mensaje)


def literal(texto):
    """Literal de texto SQL con las comillas simples duplicadas, para que sirvan rutas como `O'Connor`."""
    return "'" + str(texto).replace("'", "''") + "'"


def sql_leer_crudo(path):
    """Lee el crudo con el identificador como texto y lo convierte a BIGINT."""
    return f"""
    SELECT * REPLACE (CAST(CAST(numero_de_cliente AS DOUBLE) AS BIGINT) AS numero_de_cliente)
    FROM read_csv({literal(path)}, header = true, sample_size = -1,
                  types = {{'numero_de_cliente': 'VARCHAR', 'foto_mes': 'INTEGER'}})"""


def sql_clase(tabla):
    """Misma lógica que la cátedra: períodos consecutivos y los dos siguientes de cada cliente."""
    return f"""
    WITH p AS (
        SELECT numero_de_cliente, foto_mes, (foto_mes // 100) * 12 + foto_mes % 100 AS periodo0
        FROM {tabla}),
    l AS (
        SELECT *, lead(periodo0, 1) OVER w AS periodo1, lead(periodo0, 2) OVER w AS periodo2
        FROM p WINDOW w AS (PARTITION BY numero_de_cliente ORDER BY periodo0)),
    t AS (SELECT max(periodo0) AS ultimo FROM p)
    SELECT numero_de_cliente, foto_mes,
           CASE WHEN periodo0 < ultimo - 1 AND periodo1 = periodo0 + 1
                     AND (periodo2 IS NULL OR periodo2 > periodo0 + 2) THEN 'BAJA+2'
                WHEN periodo0 < ultimo AND (periodo1 IS NULL OR periodo1 > periodo0 + 1) THEN 'BAJA+1'
                WHEN periodo0 < ultimo - 1 THEN 'CONTINUA'
           END AS clase_ternaria
    FROM l, t"""


def regla_financiacion(marca):
    return (f"{marca}_mlimitecompra > 0 AND "
            f"{marca}_mfinanciacion_limite > {UMBRAL_FINANCIACION} * {marca}_mlimitecompra")


def regla_vencimiento(marca):
    return f"{marca}_Fvencimiento < {CENTINELA_VENCIMIENTO}"


def sql_reparar(tabla):
    """Las reparaciones de D01 sobre una tabla con las columnas del crudo."""
    reemplazos = ", ".join(
        [f"CASE WHEN {regla_financiacion(m)} THEN NULL ELSE {m}_mfinanciacion_limite END AS {m}_mfinanciacion_limite"
         for m in ("Visa", "Master")]
        + [f"CASE WHEN {regla_vencimiento(m)} THEN NULL ELSE {m}_Fvencimiento END AS {m}_Fvencimiento"
           for m in ("Visa", "Master")])
    return f"""SELECT * EXCLUDE ({", ".join(SACAR)}) REPLACE ({reemplazos}),
                      ccajas_depositos + ccajas_otras AS ccajas_depositos_otras
               FROM {tabla}"""


def huella(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def main():
    SALIDA.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute(f"CREATE TABLE texto AS SELECT numero_de_cliente FROM read_csv({literal(CRUDO.as_posix())}, header = true, all_varchar = true)")
    mal = con.execute("""SELECT count(*) FROM texto
                         WHERE try_cast(numero_de_cliente AS DOUBLE) IS NULL
                            OR CAST(numero_de_cliente AS DOUBLE) <> CAST(CAST(numero_de_cliente AS DOUBLE) AS BIGINT)""").fetchone()[0]
    exigir(mal == 0, f"{mal} identificadores no se convierten exactamente a entero")
    cientifica = con.execute("SELECT count(*) FROM texto WHERE NOT regexp_matches(numero_de_cliente, '^[0-9]+$')").fetchone()[0]
    con.execute(f"CREATE TABLE crudo AS {sql_leer_crudo(CRUDO.as_posix())}")

    tipos = dict(con.execute("SELECT column_name, column_type FROM (DESCRIBE crudo)").fetchall())
    exigir(tipos["numero_de_cliente"] == "BIGINT" and tipos["foto_mes"] == "INTEGER", tipos)
    n_crudo, n_claves, n_nulas = con.execute("""SELECT count(*), count(DISTINCT (numero_de_cliente, foto_mes)),
                                                       count(*) FILTER (WHERE numero_de_cliente IS NULL OR foto_mes IS NULL)
                                                FROM crudo""").fetchone()
    exigir(n_crudo == n_claves and n_nulas == 0, "claves cliente-mes repetidas o nulas")
    meses = [m for m, in con.execute("SELECT DISTINCT foto_mes FROM crudo ORDER BY 1").fetchall()]
    exigir(meses == MESES, f"meses inesperados: {meses}")
    columnas_crudo = list(tipos)
    exigir(all(c in columnas_crudo for c in SACAR + REPARADAS), "faltan columnas esperadas en el crudo")

    con.execute(f"CREATE TABLE clase AS {sql_clase('crudo')}")

    # Valor original de cada celda que se reemplaza por NA
    partes = []
    for marca in ("Visa", "Master"):
        partes.append(f"""SELECT numero_de_cliente, foto_mes, '{marca}_mfinanciacion_limite' AS variable,
                                 {marca}_mfinanciacion_limite AS valor_original, 'financiacion > 10 x limite de compra' AS regla
                          FROM crudo WHERE {regla_financiacion(marca)}""")
        partes.append(f"""SELECT numero_de_cliente, foto_mes, '{marca}_Fvencimiento',
                                 {marca}_Fvencimiento, 'centinela < -1000000'
                          FROM crudo WHERE {regla_vencimiento(marca)}""")
    con.execute("CREATE TABLE reparaciones AS " + " UNION ALL ".join(partes))

    con.execute(f"""
        CREATE TABLE final AS
        SELECT r.*, clase.clase_ternaria
        FROM ({sql_reparar('crudo')}) r JOIN clase USING (numero_de_cliente, foto_mes)
        ORDER BY foto_mes, numero_de_cliente""")

    # Esquema esperado
    columnas_final = [c for c, in con.execute("SELECT column_name FROM (DESCRIBE final)").fetchall()]
    esperadas = [c for c in columnas_crudo if c not in SACAR] + ["ccajas_depositos_otras", "clase_ternaria"]
    exigir(columnas_final == esperadas, "las columnas de salida no son las esperadas")
    n_final = con.execute("SELECT count(*) FROM final").fetchone()[0]
    exigir(n_final == n_crudo, (n_final, n_crudo))

    # Clase
    clase_mes = {}
    for m, c, n in con.execute("""SELECT foto_mes, coalesce(clase_ternaria, 'sin clase'), count(*)
                                  FROM final GROUP BY 1, 2 ORDER BY 1, 2""").fetchall():
        clase_mes.setdefault(m, {})[c] = n
    for m, n in ESPERADO_BAJA2.items():
        exigir(clase_mes[m].get("BAJA+2") == n, (m, clase_mes[m]))
    for m, n in ESPERADO_BAJA1.items():
        exigir(clase_mes[m].get("BAJA+1") == n, (m, clase_mes[m]))
    exigir(set(clase_mes[202108]) == {"sin clase"} and "CONTINUA" not in clase_mes[202107], "clase de julio o agosto")

    # Reparaciones
    rep = con.execute("SELECT variable, foto_mes, count(*) FROM reparaciones GROUP BY 1, 2 ORDER BY 1, 2").fetchall()
    rep_dict = {(v, m): n for v, m, n in rep}
    exigir(rep_dict[("Master_mfinanciacion_limite", 202104)] == 4159, "financiación Master de abril")
    exigir(rep_dict[("Visa_mfinanciacion_limite", 202108)] == 1035, "financiación Visa de agosto")

    # Celda por celda: fuera de las reparaciones, la salida es idéntica al crudo
    comunes = [c for c in columnas_crudo if c not in SACAR and c not in ("numero_de_cliente", "foto_mes")]
    difs = con.execute("SELECT " + ", ".join(
        f'count(*) FILTER (WHERE c."{v}" IS DISTINCT FROM f."{v}")' for v in comunes)
        + " FROM crudo c JOIN final f USING (numero_de_cliente, foto_mes)").fetchone()
    difs = dict(zip(comunes, difs))
    rep_var = {v: sum(n for (vv, _), n in rep_dict.items() if vv == v) for v in REPARADAS}
    for v, n in difs.items():
        exigir(n == rep_var.get(v, 0), f"{v}: {n} celdas distintas del crudo, se esperaban {rep_var.get(v, 0)}")
    no_nulas = con.execute("SELECT " + " + ".join(
        f"(SELECT count(*) FROM reparaciones r JOIN final f USING (numero_de_cliente, foto_mes) "
        f"WHERE r.variable = '{v}' AND f.{v} IS NOT NULL)" for v in REPARADAS)).fetchone()[0]
    exigir(no_nulas == 0, "hay celdas reparadas que no quedaron en NA")
    suma = con.execute("""SELECT count(*) FILTER (WHERE f.ccajas_depositos_otras IS DISTINCT FROM c.ccajas_depositos + c.ccajas_otras)
                          FROM crudo c JOIN final f USING (numero_de_cliente, foto_mes)""").fetchone()[0]
    exigir(suma == 0, "la suma de cajas no coincide")

    con.execute(f"COPY final TO {literal((SALIDA / 'competencia_01_d01.parquet').as_posix())} (FORMAT parquet, COMPRESSION zstd)")
    con.execute(f"""COPY (SELECT * FROM reparaciones ORDER BY foto_mes, numero_de_cliente, variable)
                    TO {literal((SALIDA / 'reparaciones_d01.parquet').as_posix())} (FORMAT parquet)""")
    resumen = {
        "decision": "D01 opcion A (5-oct-2026)",
        "versiones": {"python": platform.python_version(), "duckdb": duckdb.__version__},
        "crudo": {"archivo": CRUDO.name, "sha256": huella(CRUDO), "filas": n_crudo, "columnas": len(columnas_crudo),
                  "identificadores_en_notacion_cientifica": cientifica},
        "salida": {"filas": n_final, "columnas": len(columnas_final)},
        "columnas_quitadas": SACAR,
        "columnas_agregadas": ["ccajas_depositos_otras", "clase_ternaria"],
        "parametros": {"umbral_financiacion": UMBRAL_FINANCIACION, "centinela_vencimiento": CENTINELA_VENCIMIENTO},
        "clase_por_mes": {str(m): clase_mes[m] for m in sorted(clase_mes)},
        "reparaciones_por_variable_y_mes": {f"{v} {m}": n for (v, m), n in sorted(rep_dict.items())},
        "celdas_distintas_del_crudo": {v: n for v, n in sorted(difs.items()) if n},
    }
    (SALIDA / "resumen_d01.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
