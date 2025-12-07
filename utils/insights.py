# utils/insights.py
"""
insights.py

Provides functions to generate automated business insights from a pandas DataFrame.

Features:
- Top/Bottom performers for numeric columns
- Trend detection (increase/decrease over time)
- Anomaly detection (outliers using IQR)
- Summary statistics and distributions
- Correlation insights
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


# -------------------------
# 1. Top/Bottom Performers
# -------------------------
def top_bottom_performers(
    df: pd.DataFrame, col: str, top_n: int = 5
) -> Dict[str, Any]:
    """
    Returns the top N and bottom N values for a numeric column.

    Parameters:
    - df: input DataFrame
    - col: numeric column
    - top_n: number of rows to return

    Returns:
    - dict with 'top', 'bottom', and summary info
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ValueError(f"Column '{col}' must be numeric.")

    # Get statistics
    col_mean = df[col].mean()
    col_median = df[col].median()
    col_std = df[col].std()
    
    # Get top and bottom
    top_df = df.nlargest(top_n, col)
    bottom_df = df.nsmallest(top_n, col)

    return {
        "column": col,
        "mean": round(col_mean, 2),
        "median": round(col_median, 2),
        "std": round(col_std, 2),
        "top": top_df,
        "bottom": bottom_df,
        "top_values": top_df[col].tolist(),
        "bottom_values": bottom_df[col].tolist()
    }


# -------------------------
# 2. Trend Detection
# -------------------------
def detect_trends(df: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
    """
    Detects trends (increasing, decreasing, or stable) over time.

    Parameters:
    - df: input DataFrame
    - date_col: date column
    - value_col: numeric column

    Returns:
    - dict with trend information
    """
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError("Columns not found in DataFrame.")

    # Parse dates and values
    series_date = pd.to_datetime(df[date_col], errors="coerce")
    series_value = pd.to_numeric(df[value_col], errors="coerce")

    # Remove invalid data
    valid = ~series_date.isna() & ~series_value.isna()
    series_date = series_date[valid]
    series_value = series_value[valid]

    if len(series_date) < 2:
        return {
            "trend": "insufficient_data",
            "message": "Not enough data to detect trend.",
            "slope": 0,
            "change_percent": 0
        }

    # Sort by date
    sorted_idx = series_date.argsort()
    series_value = series_value.iloc[sorted_idx]
    series_date = series_date.iloc[sorted_idx]

    # Compute slope of linear regression
    x = np.arange(len(series_value))
    y = series_value.values
    
    if len(y) > 1:
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
    else:
        slope = 0

    # Calculate percent change from first to last
    first_val = series_value.iloc[0]
    last_val = series_value.iloc[-1]
    
    if first_val != 0:
        change_percent = ((last_val - first_val) / abs(first_val)) * 100
    else:
        change_percent = 0

    # Determine trend
    if abs(slope) < 0.01 * series_value.mean():  # Less than 1% of mean
        trend = "stable"
        message = f"'{value_col}' shows a stable trend over time."
    elif slope > 0:
        trend = "increasing"
        message = f"'{value_col}' shows an increasing trend (↗ {change_percent:.1f}% change)."
    else:
        trend = "decreasing"
        message = f"'{value_col}' shows a decreasing trend (↘ {change_percent:.1f}% change)."

    return {
        "trend": trend,
        "message": message,
        "slope": round(slope, 4),
        "change_percent": round(change_percent, 2),
        "start_date": str(series_date.iloc[0].date()),
        "end_date": str(series_date.iloc[-1].date()),
        "start_value": round(first_val, 2),
        "end_value": round(last_val, 2)
    }


# -------------------------
# 3. Anomaly Detection (IQR-based)
# -------------------------
def detect_anomalies(df: pd.DataFrame, col: str, iqr_multiplier: float = 1.5) -> Dict[str, Any]:
    """
    Returns outliers based on IQR method.

    Parameters:
    - df: input DataFrame
    - col: numeric column
    - iqr_multiplier: IQR multiplier (default 1.5)

    Returns:
    - dict with outlier information
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ValueError(f"Column '{col}' must be numeric.")

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    outlier_count = len(outliers)
    outlier_percent = (outlier_count / len(df)) * 100 if len(df) > 0 else 0

    return {
        "column": col,
        "outlier_count": outlier_count,
        "outlier_percent": round(outlier_percent, 2),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "outliers": outliers,
        "Q1": round(Q1, 2),
        "Q3": round(Q3, 2),
        "IQR": round(IQR, 2)
    }


# -------------------------
# 4. Correlation Insights
# -------------------------
def find_strong_correlations(df: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Find pairs of numeric columns with strong correlations.

    Parameters:
    - df: input DataFrame
    - threshold: correlation threshold (default 0.7)

    Returns:
    - list of correlation insights
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        return []
    
    corr_matrix = numeric_df.corr()
    
    strong_corr = []
    
    # Find strong correlations (avoid duplicates and self-correlation)
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            
            if abs(corr_val) >= threshold:
                strong_corr.append({
                    "column1": col1,
                    "column2": col2,
                    "correlation": round(corr_val, 3),
                    "strength": "strong positive" if corr_val > 0 else "strong negative"
                })
    
    # Sort by absolute correlation
    strong_corr.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    
    return strong_corr


# -------------------------
# 5. Distribution Insights
# -------------------------
def analyze_distribution(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """
    Analyze the distribution of a numeric column.

    Parameters:
    - df: input DataFrame
    - col: numeric column

    Returns:
    - dict with distribution insights
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ValueError(f"Column '{col}' must be numeric.")
    
    data = df[col].dropna()
    
    if len(data) == 0:
        return {"message": "No valid data"}
    
    # Calculate statistics
    mean = data.mean()
    median = data.median()
    std = data.std()
    skew = data.skew()
    
    # Determine skewness
    if abs(skew) < 0.5:
        skew_desc = "approximately symmetric"
    elif skew > 0:
        skew_desc = "right-skewed (positive skew)"
    else:
        skew_desc = "left-skewed (negative skew)"
    
    return {
        "column": col,
        "mean": round(mean, 2),
        "median": round(median, 2),
        "std": round(std, 2),
        "min": round(data.min(), 2),
        "max": round(data.max(), 2),
        "skewness": round(skew, 3),
        "skewness_desc": skew_desc
    }


# -------------------------
# 6. Category Insights
# -------------------------
def analyze_categories(df: pd.DataFrame, col: str, top_n: int = 10) -> Dict[str, Any]:
    """
    Analyze categorical column distribution.

    Parameters:
    - df: input DataFrame
    - col: categorical column
    - top_n: number of top categories to show

    Returns:
    - dict with category insights
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")
    
    value_counts = df[col].value_counts()
    total_count = len(df[col].dropna())
    
    top_categories = value_counts.head(top_n)
    top_percent = (top_categories / total_count * 100).round(2)
    
    return {
        "column": col,
        "unique_count": df[col].nunique(),
        "total_count": total_count,
        "most_common": value_counts.index[0] if len(value_counts) > 0 else None,
        "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
        "most_common_percent": round((value_counts.iloc[0] / total_count * 100), 2) if len(value_counts) > 0 else 0,
        "top_categories": top_categories.to_dict(),
        "top_percentages": top_percent.to_dict()
    }


# -------------------------
# 7. Generate All Insights
# -------------------------
def generate_insights(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    date_col: Optional[str] = None,
    top_n: int = 5
) -> Dict[str, Any]:
    """
    Generate comprehensive insights for the DataFrame.

    Parameters:
    - df: input DataFrame
    - numeric_cols: list of numeric columns to analyze (auto-detect if None)
    - date_col: date column for trend analysis
    - top_n: number of top/bottom values to show

    Returns:
    - dict of insights organized by category
    """
    insights = {
        "summary": {},
        "top_bottom": {},
        "trends": {},
        "anomalies": {},
        "correlations": [],
        "distributions": {},
        "categories": {}
    }
    
    # Auto-detect numeric columns if not provided
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Summary
    insights["summary"] = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "numeric_columns": len(numeric_cols),
        "missing_values": int(df.isnull().sum().sum()),
        "missing_percent": round((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2)
    }
    
    # Analyze each numeric column
    for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
        try:
            # Top/Bottom performers
            insights["top_bottom"][col] = top_bottom_performers(df, col, top_n)
            
            # Distribution
            insights["distributions"][col] = analyze_distribution(df, col)
            
            # Anomalies
            anomaly_info = detect_anomalies(df, col)
            if anomaly_info["outlier_count"] > 0:
                insights["anomalies"][col] = anomaly_info
        except Exception as e:
            print(f"Error analyzing column {col}: {e}")
    
    # Trend analysis
    if date_col and numeric_cols:
        for col in numeric_cols[:3]:  # Analyze trends for first 3 numeric columns
            try:
                trend_info = detect_trends(df, date_col, col)
                insights["trends"][col] = trend_info
            except Exception as e:
                print(f"Error detecting trend for {col}: {e}")
    
    # Correlation insights
    try:
        insights["correlations"] = find_strong_correlations(df, threshold=0.7)
    except Exception as e:
        print(f"Error finding correlations: {e}")
    
    # Category analysis
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols[:5]:  # Limit to first 5 categorical columns
        try:
            insights["categories"][col] = analyze_categories(df, col, top_n=10)
        except Exception as e:
            print(f"Error analyzing category {col}: {e}")
    
    return insights