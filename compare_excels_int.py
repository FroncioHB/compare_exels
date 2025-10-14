import re
import fnmatch
import io
import datetime as dt
from typing import List, Optional, Tuple

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

def _resolve_user_keys_or_fail(df_a: pd.DataFrame, df_b: pd.DataFrame, user_keys: List[str]) -> List[str]:
    uk_norm = [_norm_name(k) for k in user_keys]
    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)
    missing_a = [k for k in uk_norm if k not in cols_a]
    missing_b = [k for k in uk_norm if k not in cols_b]
    if missing_a or missing_b:
        msg = []
        if missing_a:
            msg.append(f"no encontradas en A: {missing_a}")
        if missing_b:
            msg.append(f"no encontradas en B: {missing_b}")
        raise ValueError(
            "Las columnas clave indicadas no existen tras normalizar nombres.\n"
            f"- Claves pedidas (normalizadas): {uk_norm}\n"
            f"- Columnas en A (normalizadas): {sorted(cols_a)}\n"
            f"- Columnas en B (normalizadas): {sorted(cols_b)}\n"
            f"- Detalle: {', '.join(msg)}"
        )
    return uk_norm

# =========================================
# Lectura & claves
# =========================================

def read_excel_as_str(file, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Lee Excel (file-like o ruta), normaliza nombres y celdas a texto (trim)."""
    df = pd.read_excel(file, sheet_name=0 if sheet_name is None else sheet_name, dtype=str)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    _normalize_columns_inplace(df)
    return df.fillna("").applymap(lambda x: x.strip() if isinstance(x, str) else x)

def pick_key_columns(df_a: pd.DataFrame, df_b: pd.DataFrame, user_keys: Optional[List[str]] = None):
    common_cols = [c for c in df_a.columns if c in df_b.columns]
    if user_keys:
        resolved = _resolve_user_keys_or_fail(df_a, df_b, user_keys)
        return resolved, "user"
    candidates = [c for c in common_cols if c in {"id", "codigo", "code", "key"}]
    if candidates:
        return [candidates[0]], "heuristic"
    for c in common_cols:
        if df_a[c].ne("").all() and df_b[c].ne("").all() and not df_a[c].duplicated().any() and not df_b[c].duplicated().any():
            return [c], "heuristic"
    return [], "index"

def _add_dup_sequence(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    df = df.copy()
    if keys:
        df["__dup_seq"] = df.groupby(keys, sort=False).cumcount()
    else:
        df["_row_"] = range(len(df))
        df["__dup_seq"] = 0
    return df

# =========================================
# Alineación (outer-merge con duplicados)
# =========================================

def align_merge_dups(df_a: pd.DataFrame, df_b: pd.DataFrame, keys: List[str]):
    a = _add_dup_sequence(df_a, keys)
    b = _add_dup_sequence(df_b, keys)
    index_cols = (keys + ["__dup_seq"]) if keys else ["_row_"]
    helper_cols = set(keys) | {"__dup_seq", "_row_"}
    value_cols = [c for c in df_a.columns if c in df_b.columns and c not in helper_cols]
    A_sub = a[index_cols + value_cols]
    B_sub = b[index_cols + value_cols]
    merged = pd.merge(A_sub, B_sub, on=index_cols, how="outer", suffixes=("_A", "_B"), sort=False)
    if keys:
        merged["dup_index"] = (merged["__dup_seq"].astype("Int64") + 1)
    return merged, value_cols, index_cols

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

def _filter_value_cols(value_cols: List[str], key_cols: List[str], ignore_patterns: List[str]) -> List[str]:
    if not ignore_patterns:
        return value_cols
    return [c for c in value_cols if not any(fnmatch.fnmatchcase(c, pat) for pat in ignore_patterns)]

# =========================================
# Construcción de hojas
# =========================================

def build_wide_from_merged(merged: pd.DataFrame, value_cols: List[str], keys: List[str],
                           tolerance: float, ignore_patterns: List[str], keep_dup_seq: bool):
    base_cols = _filter_value_cols(value_cols, keys, ignore_patterns)
    if not base_cols:
        key_part = []
        if "dup_index" in merged.columns: key_part.append("dup_index")
        if keep_dup_seq and "__dup_seq" in merged.columns: key_part.append("__dup_seq")
        key_part += keys
        return pd.DataFrame(columns=key_part), 0

    B_side = [f"{c}_B" for c in base_cols]
    A_side = [f"{c}_A" for c in base_cols]
    has_a = ~merged[A_side].isna().all(axis=1)
    has_b = ~merged[B_side].isna().all(axis=1)
    both_present = has_a & has_b
    if not both_present.any():
        key_part = []
        if "dup_index" in merged.columns: key_part.append("dup_index")
        if keep_dup_seq and "__dup_seq" in merged.columns: key_part.append("__dup_seq")
        key_part += keys
        cols = key_part[:]
        for c in base_cols:
            cols += [f"{c} (Excel A)", f"{c} (Excel B)", f"{c} (Diferencia)"]
        return pd.DataFrame(columns=cols), 0

    work = merged.loc[both_present].copy()
    A_text = work[A_side].astype(str).fillna(""); A_text.columns = base_cols
    B_text = work[B_side].astype(str).fillna(""); B_text.columns = base_cols

    A_num = A_text.applymap(_coerce_number)
    B_num = B_text.applymap(_coerce_number)
    both_num = (~A_num.isna()) & (~B_num.isna())

    diffs_mask = A_text.ne(B_text)
    if tolerance > 0:
        within_tol = (B_num - A_num).abs() <= tolerance
        diffs_mask = diffs_mask & ~(both_num & within_tol)

    diff_cells_count = int(diffs_mask.values.sum())
    if diff_cells_count == 0:
        key_part = []
        if "dup_index" in merged.columns: key_part.append("dup_index")
        if keep_dup_seq and "__dup_seq" in merged.columns: key_part.append("__dup_seq")
        key_part += keys
        cols = key_part[:]
        for c in base_cols:
            cols += [f"{c} (Excel A)", f"{c} (Excel B)", f"{c} (Diferencia)"]
        return pd.DataFrame(columns=cols), 0

    A_fmt = A_text.where(A_num.isna(), A_num.applymap(format_number_es))
    B_fmt = B_text.where(B_num.isna(), B_num.applymap(format_number_es))
    diff_num = (B_num - A_num).where(both_num)
    diff_text = diff_num.applymap(format_number_es).fillna("")
    diff_text = diff_text.where(diffs_mask, "")
    diff_text = diff_text.mask((~both_num) & diffs_mask, "DIFF")

    any_diff_row = diffs_mask.any(axis=1).to_numpy()
    key_cols_to_show = []
    if "dup_index" in work.columns: key_cols_to_show.append("dup_index")
    if keep_dup_seq and "__dup_seq" in work.columns: key_cols_to_show.append("__dup_seq")
    key_cols_to_show += keys
    keys_df = work.loc[any_diff_row, key_cols_to_show].reset_index(drop=True)

    out_parts = [keys_df]
    for c in base_cols:
        out_parts.append(A_fmt.loc[any_diff_row, [c]].rename(columns={c: f"{c} (Excel A)"}).reset_index(drop=True))
        out_parts.append(B_fmt.loc[any_diff_row, [c]].rename(columns={c: f"{c} (Excel B)"}).reset_index(drop=True))
        out_parts.append(diff_text.loc[any_diff_row, [c]].rename(columns={c: f"{c} (Diferencia)"}).reset_index(drop=True))

    wide = pd.concat(out_parts, axis=1)
    return wide, diff_cells_count

def compute_only_in_from_merged(merged: pd.DataFrame, value_cols: List[str], keys: List[str], keep_dup_seq: bool):
    A_side = [f"{c}_A" for c in value_cols]
    B_side = [f"{c}_B" for c in value_cols]
    only_in_a_mask = merged[B_side].isna().all(axis=1) & ~merged[A_side].isna().all(axis=1)
    only_in_b_mask = merged[A_side].isna().all(axis=1) & ~merged[B_side].isna().all(axis=1)

    key_cols_to_show = []
    if "dup_index" in merged.columns: key_cols_to_show.append("dup_index")
    if keep_dup_seq and "__dup_seq" in merged.columns: key_cols_to_show.append("__dup_seq")
    key_cols_to_show += keys

    if only_in_a_mask.any():
        part_keys = merged.loc[only_in_a_mask, key_cols_to_show].reset_index(drop=True)
        part_vals = merged.loc[only_in_a_mask, A_side].reset_index(drop=True); part_vals.columns = value_cols
        only_in_a = pd.concat([part_keys, part_vals], axis=1)
    else:
        only_in_a = pd.DataFrame(columns=key_cols_to_show + value_cols)

    if only_in_b_mask.any():
        part_keys = merged.loc[only_in_b_mask, key_cols_to_show].reset_index(drop=True)
        part_vals = merged.loc[only_in_b_mask, B_side].reset_index(drop=True); part_vals.columns = value_cols
        only_in_b = pd.concat([part_keys, part_vals], axis=1)
    else:
        only_in_b = pd.DataFrame(columns=key_cols_to_show + value_cols)

    return only_in_a, only_in_b

# =========================================
# Escritura a Excel (bytes para descarga)
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
st.caption("Comparador de Excel con soporte de duplicados, tolerancia y formato numérico europeo (coma).")

with st.sidebar:
    st.header("Configuración")
    file_a = st.file_uploader("Archivo A (.xlsx)", type=["xlsx"], key="file_a")
    sheet_a = st.text_input("Hoja A (opcional)", placeholder="Primera hoja por defecto")
    file_b = st.file_uploader("Archivo B (.xlsx)", type=["xlsx"], key="file_b")
    sheet_b = st.text_input("Hoja B (opcional)", placeholder="Primera hoja por defecto")

    keys_str = st.text_input("Claves (coma-separadas)", placeholder="p.ej. id, shop_id")
    ignore_str = st.text_input("Ignorar columnas (patrones)", placeholder="p.ej. created_at,fecha_*,hash")
    tol = st.text_input("Tolerancia numérica", value="0.0", help="Usa coma o punto. Se interpreta como absoluto.")
    keep_dup_seq = st.checkbox("Exportar __dup_seq (diagnóstico)", value=False)

    run = st.button("🔍 Comparar")

if run:
    try:
        if not (file_a and file_b):
            st.warning("Sube **Archivo A** y **Archivo B** para comenzar.")
            st.stop()

        # Tolerancia: admitir coma o punto
        try:
            tolerance = float((tol or "0").replace(",", "."))
        except ValueError:
            st.error("La tolerancia debe ser un número. Ej: 0.01 o 0,01")
            st.stop()

        # Leer excels
        df_a = read_excel_as_str(file_a, sheet_a or None)
        df_b = read_excel_as_str(file_b, sheet_b or None)

        # Resolver claves
        user_keys = [k.strip() for k in (keys_str or "").split(",") if k.strip()] or None
        keys, mode = pick_key_columns(df_a, df_b, user_keys)
        st.info(f"Claves usadas: **{keys if keys else '(índice por posición)'}**  ·  modo: **{mode}**")

        # Alinear
        merged, value_cols, index_cols = align_merge_dups(df_a, df_b, keys)

        ignore_patterns = [p.strip() for p in (ignore_str or "").split(",") if p.strip()]

        # Construir hojas
        only_in_a, only_in_b = compute_only_in_from_merged(merged, value_cols, keys, keep_dup_seq=keep_dup_seq)
        comparison_wide, diff_cells_count = build_wide_from_merged(
            merged, value_cols, keys, tolerance, ignore_patterns, keep_dup_seq=keep_dup_seq
        )

        # Métricas
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Filas solo en A", len(only_in_a))
        c2.metric("Filas solo en B", len(only_in_b))
        c3.metric("Celdas distintas", diff_cells_count)
        c4.metric("Columnas comparadas", len([c for c in value_cols if not ignore_patterns or all(not fnmatch.fnmatchcase(c, p) for p in ignore_patterns)]))

        # Previsualización
        st.subheader("comparison_wide")
        st.dataframe(comparison_wide, use_container_width=True, height=360)
        with st.expander("only_in_file_a"):
            st.dataframe(only_in_a, use_container_width=True, height=260)
        with st.expander("only_in_file_b"):
            st.dataframe(only_in_b, use_container_width=True, height=260)

        # Generar Excel + descarga
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
