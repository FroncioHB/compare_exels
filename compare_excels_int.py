import re
import fnmatch
import io
import datetime as dt
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

# =========================================
# Normalización de nombres
# =========================================

def _norm_name(s: str) -> str:
    if s is None:
        return ""
    t = str(s).replace("\u00A0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower()

def _normalize_columns_inplace(df: pd.DataFrame) -> None:
    df.columns = [_norm_name(c) for c in df.columns]

# =========================================
# Lectura & cabeceras
# =========================================

def read_excel_as_str(file, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Lee Excel (file-like o ruta), normaliza nombres y celdas a texto (trim)."""
    df = pd.read_excel(file, sheet_name=0 if sheet_name is None else sheet_name, dtype=str)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    _normalize_columns_inplace(df)
    return df.fillna("").applymap(lambda x: x.strip() if isinstance(x, str) else x)

def read_excel_header_columns(file, sheet_name: Optional[str] = None) -> List[str]:
    """Lee SOLO la cabecera (nombres originales) para poblar listas del UI."""
    df_head = pd.read_excel(file, sheet_name=0 if sheet_name is None else sheet_name, dtype=str, nrows=0)
    if isinstance(df_head, pd.Series):
        df_head = df_head.to_frame()
    return list(df_head.columns)

# =========================================
# Duplicados: helper
# =========================================

def _add_dup_sequence(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    df = df.copy()
    if keys:
        df["__dup_seq"] = df.groupby(keys, sort=False).cumcount()
    else:
        df["_row_"] = range(len(df))
        df["__dup_seq"] = 0
    return df

# =========================================
# Alineación (outer-merge con duplicados) – universal (incluye TODAS las columnas de A y B)
# =========================================

def align_merge_dups(df_a: pd.DataFrame, df_b: pd.DataFrame, keys: List[str]):
    """
    Devuelve un merged con:
      - claves (y __dup_seq / _row_)
      - TODAS las columnas de A renombradas con sufijo _A
      - TODAS las columnas de B renombradas con sufijo _B
    Así luego podemos comparar pares arbitrarios (nombres distintos entre A y B).
    """
    a = _add_dup_sequence(df_a, keys)
    b = _add_dup_sequence(df_b, keys)

    index_cols = (keys + ["__dup_seq"]) if keys else ["_row_"]
    helper_cols = set(keys) | {"__dup_seq", "_row_"}

    a_vals = [c for c in df_a.columns if c not in helper_cols]
    b_vals = [c for c in df_b.columns if c not in helper_cols]

    A_sub = a[index_cols + a_vals].copy()
    B_sub = b[index_cols + b_vals].copy()

    # Renombrar valores con sufijos (evita colisiones de nombres)
    A_sub.rename(columns={c: f"{c}_A" for c in a_vals}, inplace=True)
    B_sub.rename(columns={c: f"{c}_B" for c in b_vals}, inplace=True)

    merged = pd.merge(A_sub, B_sub, on=index_cols, how="outer", sort=False)

    if keys:
        merged["dup_index"] = (merged["__dup_seq"].astype("Int64") + 1)

    return merged, a_vals, b_vals, index_cols

# =========================================
# Parseo & formato ES
# =========================================

_num_point_decimal_regex = re.compile(r"-?\d+\.\d+")

def _coerce_number(s):
    if s is None:
        return np.nan
    t = str(s).strip().replace("\u00A0", "").replace(" ", "")
    if t == "":
        return np.nan
    if "," in t:
        t = t.replace(".", "").replace(",", ".")
        return pd.to_numeric(t, errors="coerce")
    if _num_point_decimal_regex.fullmatch(t):
        return pd.to_numeric(t, errors="coerce")
    t = t.replace(".", "")
    return pd.to_numeric(t, errors="coerce")

def format_number_es(x, max_decimals=6):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if float(xf).is_integer():
        return str(int(xf))
    s = f"{xf:.{max_decimals}f}".rstrip("0").rstrip(".")
    return s.replace(".", ",")

def _format_cell_as_text_es(v):
    n = _coerce_number(v)
    if pd.isna(n):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        return str(v)
    return format_number_es(n)

def format_df_all_cells_es(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.applymap(_format_cell_as_text_es).astype("string")

# =========================================
# Utils
# =========================================

def _filter_value_cols(value_cols: List[str], ignore_patterns: List[str]) -> List[str]:
    if not ignore_patterns:
        return value_cols
    return [c for c in value_cols if not any(fnmatch.fnmatchcase(c, pat) for pat in ignore_patterns)]

# =========================================
# Construcción de hojas a partir de PARES (A_col_norm, B_col_norm)
# =========================================

def build_wide_from_pairs(
    merged: pd.DataFrame,
    pairs: List[Tuple[str, str]],   # [(a_norm, b_norm), ...]
    keys: List[str],
    tolerance: float,
    keep_dup_seq: bool
) -> Tuple[pd.DataFrame, int]:
    """
    Usa pares A↔B explícitos para construir la hoja 'comparison_wide'.
    """
    if not pairs:
        cols = []
        if "dup_index" in merged.columns: cols.append("dup_index")
        if keep_dup_seq and "__dup_seq" in merged.columns: cols.append("__dup_seq")
        cols += keys
        return pd.DataFrame(columns=cols), 0

    # Filas comparables: donde hay A y B al menos en ALGUNA de las columnas pareadas
    # (para el filtro de filas con diferencia trabajaremos por-par luego).
    diff_cells_total = 0
    parts = []

    # Claves a mostrar
    key_cols = []
    if "dup_index" in merged.columns: key_cols.append("dup_index")
    if keep_dup_seq and "__dup_seq" in merged.columns: key_cols.append("__dup_seq")
    key_cols += keys

    # Empezamos con DataFrame vacío de claves; lo crearemos cuando haya difs
    keys_block = None

    # Recorremos pares y vamos apilando las 3 columnas por par
    rows_mask_anydiff = None
    col_blocks = []

    for (a_col, b_col) in pairs:
        a_series = merged.get(f"{a_col}_A")
        b_series = merged.get(f"{b_col}_B")

        # Si alguna de las series no existe, saltamos ese par silenciosamente
        if a_series is None or b_series is None:
            continue

        A_text = a_series.astype(str).fillna("")
        B_text = b_series.astype(str).fillna("")

        A_num = A_text.map(_coerce_number)
        B_num = B_text.map(_coerce_number)
        both_num = (~A_num.isna()) & (~B_num.isna())

        diff_num = (B_num - A_num).where(both_num)
        within_tol = (diff_num.abs() <= tolerance) if tolerance > 0 else pd.Series(False, index=diff_num.index)

        diffs_mask = (A_text != B_text)
        if tolerance > 0:
            diffs_mask = diffs_mask & ~(both_num & within_tol)

        diff_cells = int(diffs_mask.sum())
        diff_cells_total += diff_cells

        # Formateo de salida
        A_fmt = A_text.where(A_num.isna(), A_num.map(format_number_es))
        B_fmt = B_text.where(B_num.isna(), B_num.map(format_number_es))
        diff_text = diff_num.map(format_number_es).fillna("")
        diff_text = diff_text.where(diffs_mask, "")
        diff_text = diff_text.mask((~both_num) & diffs_mask, "DIFF")

        # Acumular máscara de filas con alguna diferencia
        rows_mask_anydiff = diffs_mask if rows_mask_anydiff is None else (rows_mask_anydiff | diffs_mask)

        # Nombres de columnas de salida (mostramos los nombres normalizados; si quieres originales, podemos mapear)
        a_lbl = f"{a_col} (Excel A)"
        b_lbl = f"{b_col} (Excel B)"
        d_lbl = f"{a_col}↔{b_col} (Diferencia)"

        col_blocks.append(
            pd.DataFrame({a_lbl: A_fmt, b_lbl: B_fmt, d_lbl: diff_text})
        )

    # Si no hubo pares válidos o no hay diferencias, devolvemos cabecera
    if not col_blocks:
        cols = key_cols[:]
        return pd.DataFrame(columns=cols), 0

    # Filtramos solo filas con alguna diferencia en cualquiera de los pares
    if rows_mask_anydiff is None or not rows_mask_anydiff.any():
        # solo cabecera con columnas (sin filas)
        header_cols = key_cols[:]
        for block in col_blocks:
            header_cols += list(block.columns)
        return pd.DataFrame(columns=header_cols), 0

    # Construimos bloque de claves para las filas con diferencias
    keys_block = merged.loc[rows_mask_anydiff, key_cols].reset_index(drop=True)

    # Concatenamos columnas de todos los pares, ya filtradas
    cols_concat = [blk.loc[rows_mask_anydiff].reset_index(drop=True) for blk in col_blocks]
    wide = pd.concat([keys_block] + cols_concat, axis=1)
    return wide, diff_cells_total

def compute_only_in_from_merged(
    merged: pd.DataFrame,
    a_values: List[str],
    b_values: List[str],
    keys: List[str],
    keep_dup_seq: bool
):
    """
    Construye 'only_in_file_a' con claves + TODAS las columnas de A seleccionadas,
    y 'only_in_file_b' con claves + TODAS las columnas de B seleccionadas.
    """
    A_side = [f"{c}_A" for c in a_values]
    B_side = [f"{c}_B" for c in b_values]

    # Fila solo en A: no hay ninguna B y sí hay algo en A
    only_in_a_mask = merged[B_side].isna().all(axis=1) & ~merged[A_side].isna().all(axis=1)
    only_in_b_mask = merged[A_side].isna().all(axis=1) & ~merged[B_side].isna().all(axis=1)

    key_cols = []
    if "dup_index" in merged.columns: key_cols.append("dup_index")
    if keep_dup_seq and "__dup_seq" in merged.columns: key_cols.append("__dup_seq")
    key_cols += keys

    # A
    if only_in_a_mask.any():
        part_keys = merged.loc[only_in_a_mask, key_cols].reset_index(drop=True)
        part_vals = merged.loc[only_in_a_mask, A_side].reset_index(drop=True)
        part_vals.columns = a_values
        only_in_a = pd.concat([part_keys, part_vals], axis=1)
    else:
        only_in_a = pd.DataFrame(columns=key_cols + a_values)

    # B
    if only_in_b_mask.any():
        part_keys = merged.loc[only_in_b_mask, key_cols].reset_index(drop=True)
        part_vals = merged.loc[only_in_b_mask, B_side].reset_index(drop=True)
        part_vals.columns = b_values
        only_in_b = pd.concat([part_keys, part_vals], axis=1)
    else:
        only_in_b = pd.DataFrame(columns=key_cols + b_values)

    return only_in_a, only_in_b

# =========================================
# Excel (bytes para descarga)
# =========================================

def build_workbook_bytes(comparison_wide: pd.DataFrame, only_in_a: pd.DataFrame, only_in_b: pd.DataFrame) -> bytes:
    cw_es = format_df_all_cells_es(comparison_wide)
    only_a_es = format_df_all_cells_es(only_in_a)
    only_b_es = format_df_all_cells_es(only_in_b)
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        cw_es.to_excel(writer, index=False, sheet_name="comparison_wide")
        only_a_es.to_excel(writer, index=False, sheet_name="only_in_file_a")
        only_b_es.to_excel(writer, index=False, sheet_name="only_in_file_b")
    bio.seek(0)
    return bio.read()

# =========================================
# UI Streamlit
# =========================================

st.set_page_config(page_title="Excel Diff (ES)", page_icon="📊", layout="wide")
st.title("📊 Excel Diff (ES)")
st.caption("Comparador de Excel con mapeo de claves y emparejamiento de columnas A↔B (1:1 por orden o manual).")

with st.sidebar:
    st.header("Archivos")
    file_a = st.file_uploader("Archivo A (.xlsx)", type=["xlsx"], key="file_a")
    sheet_a = st.text_input("Hoja A (opcional)", placeholder="Primera hoja por defecto")
    file_b = st.file_uploader("Archivo B (.xlsx)", type=["xlsx"], key="file_b")
    sheet_b = st.text_input("Hoja B (opcional)", placeholder="Primera hoja por defecto")

    # Cabeceras originales para UI
    cols_a_original = []
    cols_b_original = []
    if file_a:
        try:
            cols_a_original = read_excel_header_columns(file_a, sheet_a or None)
        except Exception as e:
            st.warning(f"No se pudo leer la cabecera de A: {e}")
    if file_b:
        try:
            cols_b_original = read_excel_header_columns(file_b, sheet_b or None)
        except Exception as e:
            st.warning(f"No se pudo leer la cabecera de B: {e}")

    st.header("Claves (A → B)")
    keys_orig_a = st.multiselect(
        "Columnas clave de A",
        options=cols_a_original,
        default=[],
        help="Selecciona una o varias columnas de A que formarán la clave compuesta."
    )

    key_map: Dict[str, str] = {}
    if keys_orig_a:
        st.markdown("**Mapeo de claves A → B**")
        for a_col in keys_orig_a:
            b_choice = st.selectbox(
                f"Columna equivalente en B para A: {a_col}",
                options=(cols_b_original or [""]),
                index=(cols_b_original.index(a_col) if (a_col in cols_b_original) else 0) if cols_b_original else 0,
                key=f"mapkey_{a_col}"
            )
            if b_choice:
                key_map[a_col] = b_choice

    st.header("Emparejamiento de columnas de VALORES")
    mode_auto_pairs = st.checkbox("Comparar columnas 1↔1 por orden (omitir sobrantes)", value=True)

    # Controles del modo manual (solo visibles si se desmarca)
    manual_pairs: List[Tuple[str, str]] = []
    selected_a_for_pairs = []
    if not mode_auto_pairs:
        selected_a_for_pairs = st.multiselect(
            "Selecciona columnas de A a comparar (no-clave)",
            options=[c for c in cols_a_original if _norm_name(c) not in [_norm_name(k) for k in keys_orig_a]],
            default=[],
            help="Solo se compararán las columnas de A que elijas aquí."
        )
        st.markdown("**Empareja cada columna de A con su correspondiente en B**")
        for a_col in selected_a_for_pairs:
            b_choice = st.selectbox(
                f"B para A:{a_col}",
                options=["— Omitir —"] + cols_b_original,
                index=0,
                key=f"pair_{a_col}"
            )
            if b_choice and b_choice != "— Omitir —":
                manual_pairs.append((_norm_name(a_col), _norm_name(b_choice)))

    st.header("Parámetros")
    ignore_str = st.text_input("Ignorar columnas (patrones)", placeholder="p.ej. created_at,fecha_*,hash")
    tol = st.text_input("Tolerancia numérica", value="0.0", help="Usa coma o punto. Se interpreta como absoluto.")
    keep_dup_seq = st.checkbox("Exportar __dup_seq (diagnóstico)", value=False)

    run = st.button("🔍 Comparar")

# --- Ejecución ---
if run:
    try:
        if not (file_a and file_b):
            st.warning("Sube **Archivo A** y **Archivo B** para comenzar.")
            st.stop()

        # Validación claves
        if not keys_orig_a:
            st.error("Selecciona al menos **una** columna clave de A.")
            st.stop()
        if not key_map or any(k not in key_map for k in keys_orig_a):
            st.error("Debes indicar la columna equivalente en B para **cada** clave de A.")
            st.stop()

        # Tolerancia
        try:
            tolerance = float((tol or "0").replace(",", "."))
        except ValueError:
            st.error("La tolerancia debe ser un número. Ej: 0.01 o 0,01")
            st.stop()

        # Leer excels normalizados
        df_a = read_excel_as_str(file_a, sheet_a or None)
        df_b = read_excel_as_str(file_b, sheet_b or None)

        # Normalizar claves A y sus equivalentes B
        keys_norm_a = [_norm_name(k) for k in keys_orig_a]
        keys_norm_b_map = { _norm_name(a): _norm_name(b) for a, b in key_map.items() }

        # Comprobación existencia de las claves en A y B
        missing_in_a = [k for k in keys_norm_a if k not in df_a.columns]
        missing_in_b = [keys_norm_b_map[k] for k in keys_norm_a if keys_norm_b_map[k] not in df_b.columns]
        if missing_in_a:
            st.error(f"Estas claves no existen en A (tras normalizar): {missing_in_a}")
            st.stop()
        if missing_in_b:
            st.error(f"Estas columnas mapeadas no existen en B (tras normalizar): {missing_in_b}")
            st.stop()

        # Crear copia de B con columnas de clave con los NOMBRES de A
        df_b_mapped = df_b.copy()
        for a_norm in keys_norm_a:
            df_b_mapped[a_norm] = df_b_mapped[keys_norm_b_map[a_norm]]

        # Alinear (incluye TODAS las columnas de A y B)
        merged, a_values_all, b_values_all, index_cols = align_merge_dups(df_a, df_b_mapped, keys_norm_a)

        # --- Construcción de PARES de comparación ---
        pairs: List[Tuple[str, str]] = []

        if mode_auto_pairs:
            # Por orden: todas las no-clave en A y B
            helper_cols = set(keys_norm_a) | {"__dup_seq", "_row_"}
            a_values = [c for c in a_values_all if c not in helper_cols]
            b_values = [c for c in b_values_all if c not in helper_cols]

            # Aplicar patrones de ignorar sobre los nombres NORMALIZADOS
            ignore_patterns = [p.strip() for p in (ignore_str or "").split(",") if p.strip()]
            if ignore_patterns:
                a_values = _filter_value_cols(a_values, ignore_patterns)
                b_values = _filter_value_cols(b_values, ignore_patterns)

            n = min(len(a_values), len(b_values))
            omitted_a = max(0, len(a_values) - n)
            omitted_b = max(0, len(b_values) - n)

            if omitted_a or omitted_b:
                st.warning(f"Modo 1↔1 por orden: omitidas {omitted_a} col(s) de A y {omitted_b} col(s) de B por longitud distinta.")

            pairs = list(zip(a_values[:n], b_values[:n]))

            # Para 'only_in_*' usaremos SOLO las columnas efectivamente emparejadas
            a_values_for_only = [a for a, _ in pairs]
            b_values_for_only = [b for _, b in pairs]
        else:
            # Modo manual: usar solo los pares seleccionados
            if not manual_pairs:
                st.error("No has emparejado ninguna columna de valores. Añade al menos un par A↔B o activa el modo 1↔1.")
                st.stop()

            # También aplicamos 'ignore' por seguridad
            ignore_patterns = [p.strip() for p in (ignore_str or "").split(",") if p.strip()]
            if ignore_patterns:
                manual_pairs = [
                    (a, b) for (a, b) in manual_pairs
                    if all(not fnmatch.fnmatchcase(a, pat) for pat in ignore_patterns)
                    and all(not fnmatch.fnmatchcase(b, pat) for pat in ignore_patterns)
                ]
                if not manual_pairs:
                    st.error("Todos los pares fueron excluidos por los patrones de ignorar. Ajusta tu selección.")
                    st.stop()

            # Validar existencia real en merged (por si nombres raros)
            valid_pairs = []
            for a, b in manual_pairs:
                if f"{a}_A" in merged.columns and f"{b}_B" in merged.columns:
                    valid_pairs.append((a, b))
            if not valid_pairs:
                st.error("Ningún par A↔B es válido tras normalizar. Revisa nombres.")
                st.stop()
            pairs = valid_pairs

            a_values_for_only = [a for a, _ in pairs]
            b_values_for_only = [b for _, b in pairs]

        # Construir hojas
        comparison_wide, diff_cells_count = build_wide_from_pairs(
            merged, pairs, keys_norm_a, tolerance, keep_dup_seq=keep_dup_seq
        )
        only_in_a, only_in_b = compute_only_in_from_merged(
            merged, a_values_for_only, b_values_for_only, keys_norm_a, keep_dup_seq=keep_dup_seq
        )

        # Métricas
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Filas solo en A", len(only_in_a))
        c2.metric("Filas solo en B", len(only_in_b))
        c3.metric("Celdas distintas", diff_cells_count)
        c4.metric("Pares comparados", len(pairs))

        # Previsualización
        st.subheader("comparison_wide")
        st.dataframe(comparison_wide, use_container_width=True, height=360)
        with st.expander("only_in_file_a"):
            st.dataframe(only_in_a, use_container_width=True, height=260)
        with st.expander("only_in_file_b"):
            st.dataframe(only_in_b, use_container_width=True, height=260)

        # Descarga
        wb_bytes = build_workbook_bytes(comparison_wide, only_in_a, only_in_b)
        default_name = f"compare_result_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        st.download_button(
            "💾 Descargar Excel",
            data=wb_bytes,
            file_name=default_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)
