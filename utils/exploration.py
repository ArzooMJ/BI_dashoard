# utils/exploration.py

import pandas as pd
import numpy as np

def get_basic_info(df: pd.DataFrame) -> dict:
    """
    Returns basic dataset information: shape, columns, and dtypes.
    """
    info = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict()
    }
    return info


def get_summary_statistics(df: pd.DataFrame) -> dict:
    """
    Computes summary statistics for numerical and categorical columns.
    """
    summary = {}

    # Numerical summary
    num_cols = df.select_dtypes(include=[np.number]).columns
    summary["numerical"] = df[num_cols].describe().to_dict() if len(num_cols) > 0 else {}

    # Adding median manually because describe() does not include it by default
    if len(num_cols) > 0:
        medians = df[num_cols].median().to_dict()
        summary["numerical"]["median"] = medians

    # Categorical summary
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    cat_summary = {}

    for col in cat_cols:
        cat_summary[col] = {
            "unique_values": df[col].nunique(),
            "most_frequent": df[col].mode()[0] if df[col].mode().size > 0 else None,
            "top_5_value_counts": df[col].value_counts().head(5).to_dict()
        }

    summary["categorical"] = cat_summary

    return summary


def get_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame showing missing counts and percentages per column.
    """
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100

    report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percent": missing_percent.round(2)
    })

    return report


def get_outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers using the IQR method for numerical columns.
    Returns counts of outliers per column.
    """
    num_cols = df.select_dtypes(include=[np.number])
    outlier_counts = {}

    for col in num_cols.columns:
        Q1 = num_cols[col].quantile(0.25)
        Q3 = num_cols[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = num_cols[(num_cols[col] < lower_bound) | (num_cols[col] > upper_bound)]
        outlier_counts[col] = outliers.shape[0]

    return pd.DataFrame.from_dict(outlier_counts, orient="index", columns=["outlier_count"])


def get_sample_rows(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Returns the first n rows of the dataset.
    """
    return df.head(n)
