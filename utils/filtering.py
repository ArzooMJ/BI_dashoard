"""
utils/filtering.py

Provides helpers for the Gradio UI:
- column metadata detection
- numeric bounds and unique-value helpers
- categorical and date option helpers
- robust apply_all_filters() that returns (filtered_df, row_count)

Designed to be compatible with your app.py callbacks which expect:
- get_column_metadata(df)
- get_numeric_bounds(df, col)
- get_numeric_unique_values(df, col, max_values=1000)
- get_categorical_options(df, col, max_options=500)
- get_date_bounds(df, col)
- get_date_unique_values(df, col, max_values=1000)
- apply_all_filters(df, num_filters, cat_filters, date_filters, display_columns=None)
"""

from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
import warnings

# Suppress pandas UserWarning about ambiguous date parsing in UI helper (we handle gracefully)
warnings.filterwarnings("ignore", message="Could not infer format*")

# -----------------------------
# 1. Column metadata
# -----------------------------
def get_column_metadata(df: pd.DataFrame, max_unique_for_categorical: int = 200) -> Dict[str, Any]:
    """
    Returns a dict describing each column:
      {
        "numeric": [col1, ...],
        "categorical": [colA, ...],
        "date": [date_col1, ...],
        "meta": {
            col: {"type": "numeric"/"categorical"/"date", ...}
        }
      }
    The `meta` mapping contains additional info for each column.
    """
    if df is None:
        return {}

    meta_map = {}
    numerical = []
    categorical = []
    datecols = []

    for col in df.columns:
        series = df[col]
        # Numeric detection
        if pd.api.types.is_numeric_dtype(series):
            numerical.append(col)
            non_na = pd.to_numeric(series, errors="coerce").dropna()
            if non_na.empty:
                meta_map[col] = {"type": "numeric", "min": None, "max": None, "n_unique": 0}
            else:
                meta_map[col] = {
                    "type": "numeric",
                    "min": float(non_na.min()),
                    "max": float(non_na.max()),
                    "n_unique": int(non_na.nunique()),
                }
            continue

        # Date detection
        # If dtype is datetime already
        if pd.api.types.is_datetime64_any_dtype(series):
            datecols.append(col)
            non_na = pd.to_datetime(series, errors="coerce").dropna()
            if non_na.empty:
                meta_map[col] = {"type": "date", "min": None, "max": None, "n_unique": 0}
            else:
                meta_map[col] = {
                    "type": "date",
                    "min": non_na.min().strftime("%Y-%m-%d"),
                    "max": non_na.max().strftime("%Y-%m-%d"),
                    "n_unique": int(non_na.nunique()),
                }
            continue

        # Try to parse a small sample for datetime-like strings
        sample = series.dropna().astype(str).head(20)
        if len(sample) > 0:
            parsed = pd.to_datetime(sample, errors="coerce")
            # if a substantial fraction parses, treat as date
            if parsed.notna().sum() >= max(1, int(0.6 * len(parsed))):
                datecols.append(col)
                parsed_full = pd.to_datetime(series, errors="coerce").dropna()
                if parsed_full.empty:
                    meta_map[col] = {"type": "date", "min": None, "max": None, "n_unique": 0}
                else:
                    meta_map[col] = {
                        "type": "date",
                        "min": parsed_full.min().strftime("%Y-%m-%d"),
                        "max": parsed_full.max().strftime("%Y-%m-%d"),
                        "n_unique": int(parsed_full.nunique()),
                    }
                continue

        # fallback: categorical
        categorical.append(col)
        uniques = series.dropna().unique().tolist()
        n_unique = len(uniques)
        if n_unique > max_unique_for_categorical:
            uniques = uniques[:max_unique_for_categorical]
        # convert Numpy types to Python-native for JSON/Gradio
        uniques = [_convert(v) for v in uniques]
        meta_map[col] = {"type": "categorical", "values": uniques, "n_unique": n_unique}

    return {
        "numeric": numerical,
        "categorical": categorical,
        "date": datecols,
        "meta": meta_map,
    }


def _convert(v):
    # convert numpy scalars to Python scalars
    if isinstance(v, (np.generic,)):
        return v.item()
    return v


# -----------------------------
# 2. Numeric helpers
# -----------------------------
def get_numeric_bounds(df: pd.DataFrame, col: str) -> Tuple[Optional[float], Optional[float], float]:
    """
    Return (min, max, step) for a numeric column.
    - step chosen as 1 for integer-like ranges or (max-min)/100 otherwise.
    """
    if df is None or col not in df.columns:
        return None, None, 1.0
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return None, None, 1.0
    col_min = float(series.min())
    col_max = float(series.max())
    # determine if integers
    if np.all(np.mod(series.dropna(), 1) == 0):
        step = 1
    else:
        span = col_max - col_min
        step = float(span / 100) if span > 0 else 1.0
    return col_min, col_max, step


def get_numeric_unique_values(df: pd.DataFrame, col: str, max_values: int = 1000) -> List[Any]:
    """
    Return sorted unique values for numeric column.
    If there are more than max_values uniques, return a linspace of max_values values.
    """
    if df is None or col not in df.columns:
        return []
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return []
    uniques = np.sort(series.unique())
    if len(uniques) <= max_values:
        return [_convert(v) for v in uniques.tolist()]
    # return representative values
    vals = np.linspace(series.min(), series.max(), num=max_values)
    return [_convert(v) for v in vals.tolist()]


# -----------------------------
# 3. Categorical helpers
# -----------------------------
def get_categorical_options(df: pd.DataFrame, col: str, max_options: int = 500) -> List[str]:
    if df is None or col not in df.columns:
        return []
    series = df[col].dropna()
    uniques = series.unique().tolist()
    # cap
    if len(uniques) > max_options:
        uniques = uniques[:max_options]
    return [str(_convert(v)) for v in uniques]


# -----------------------------
# 4. Date helpers
# -----------------------------
def get_date_bounds(df: pd.DataFrame, col: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (min_date_str, max_date_str) in YYYY-MM-DD format for a date-like column.
    """
    if df is None or col not in df.columns:
        return None, None
    # try parsing entire column robustly
    parsed = pd.to_datetime(df[col], errors="coerce")
    parsed = parsed.dropna()
    if parsed.empty:
        return None, None
    min_date = parsed.min().strftime("%Y-%m-%d")
    max_date = parsed.max().strftime("%Y-%m-%d")
    return min_date, max_date


def get_date_unique_values(df: pd.DataFrame, col: str, max_values: int = 1000) -> List[str]:
    """
    Return unique date strings in YYYY-MM-DD; capped by max_values.
    """
    if df is None or col not in df.columns:
        return []
    parsed = pd.to_datetime(df[col], errors="coerce").dropna()
    if parsed.empty:
        return []
    uniques = pd.Series(parsed.dt.strftime("%Y-%m-%d").unique())
    uniques = uniques.sort_values()
    if len(uniques) <= max_values:
        return uniques.tolist()
    # sample evenly if too many
    idx = np.linspace(0, len(uniques) - 1, num=max_values).astype(int)
    return uniques.iloc[idx].tolist()


# -----------------------------
# 5. Apply all filters
# -----------------------------
def apply_all_filters(
    df: pd.DataFrame,
    num_filters: Dict[str, Dict[str, Any]] = None,
    cat_filters: Dict[str, List[Any]] = None,
    date_filters: Dict[str, Dict[str, Any]] = None,
    display_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Apply numeric, categorical and date filters to df.
    num_filters example: {"age": {"min": 20, "max": 50}}
    cat_filters example: {"city": ["NY", "LA"]}
    date_filters example: {"order_date": {"start": "2020-01-01", "end": "2020-12-31"}}
    display_columns: optional list to subset the returned dataframe columns.
    Returns (filtered_df, row_count).
    """
    if df is None:
        return pd.DataFrame(), 0
    if df.empty:
        return df, 0

    filtered = df.copy()

    # numeric
    if num_filters:
        for col, crit in num_filters.items():
            if col not in filtered.columns:
                continue
            try:
                series = pd.to_numeric(filtered[col], errors="coerce")
                if "min" in crit and crit["min"] is not None:
                    filtered = filtered[series >= crit["min"]]
                    series = pd.to_numeric(filtered[col], errors="coerce")
                if "max" in crit and crit["max"] is not None:
                    filtered = filtered[series <= crit["max"]]
                    series = pd.to_numeric(filtered[col], errors="coerce")
                # optional exact-values filter
                if "values" in crit and crit["values"]:
                    # ensure comparables
                    vals = [float(v) for v in crit["values"]]
                    filtered = filtered[filtered[col].astype(float).isin(vals)]
            except Exception:
                # skip invalid numeric filter
                continue

    # categorical
    if cat_filters:
        for col, vals in cat_filters.items():
            if col not in filtered.columns:
                continue
            if not vals:
                continue
            # compare as strings (so dates-as-strings or numbers-as-strings work)
            filtered = filtered[filtered[col].astype(str).isin([str(v) for v in vals])]

    # date filters
    if date_filters:
        for col, crit in date_filters.items():
            if col not in filtered.columns:
                continue
            # try parse column to datetime
            parsed = pd.to_datetime(filtered[col], errors="coerce")
            if parsed.isna().all():
                # fallback: compare string membership if crit provides strings
                if isinstance(crit, dict) and "values" in crit and crit["values"]:
                    filtered = filtered[filtered[col].astype(str).isin([str(v) for v in crit["values"]])]
                continue
            # if start/end provided
            start = crit.get("start") if isinstance(crit, dict) else None
            end = crit.get("end") if isinstance(crit, dict) else None
            values = crit.get("values") if isinstance(crit, dict) else None

            if start:
                try:
                    s = pd.to_datetime(start, errors="coerce")
                    if not pd.isna(s):
                        filtered = filtered[parsed >= s]
                        parsed = pd.to_datetime(filtered[col], errors="coerce")
                except Exception:
                    pass
            if end:
                try:
                    e = pd.to_datetime(end, errors="coerce")
                    if not pd.isna(e):
                        filtered = filtered[parsed <= e]
                        parsed = pd.to_datetime(filtered[col], errors="coerce")
                except Exception:
                    pass
            # exact membership (dates provided as YYYY-MM-DD strings)
            if values:
                vals_parsed = pd.to_datetime(pd.Series(values), errors="coerce").dt.strftime("%Y-%m-%d").dropna().tolist()
                # ensure column as YYYY-MM-DD strings
                col_str = pd.to_datetime(filtered[col], errors="coerce").dt.strftime("%Y-%m-%d")
                filtered = filtered[col_str.isin(vals_parsed)]

    # subset columns if requested
    if display_columns:
        keep = [c for c in display_columns if c in filtered.columns]
        try:
            filtered = filtered[keep]
        except Exception:
            pass

    row_count = len(filtered)
    return filtered, row_count
