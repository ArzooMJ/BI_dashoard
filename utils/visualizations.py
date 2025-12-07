# utils/visualizations.py
"""
Enhanced visualization helpers using Plotly.

Provides:
- make_time_series(...) - Time series with optional grouping
- make_distribution(...) - Histogram with box plot
- make_category_chart(...) - Bar/Pie charts with aggregations
- make_scatter_or_heatmap(...) - Scatter plots and correlation heatmaps
- fig_to_png_bytes(...) - Export to PNG
"""

from typing import Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# -------------------------
# Helpers
# -------------------------
def _ensure_column_exists(df: pd.DataFrame, col: str, allow_none: bool = False):
    """Validate that a column exists in the DataFrame."""
    if allow_none and (col is None or col == ""):
        return
    if col is None or col == "":
        raise ValueError("Column name must be provided.")
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")


# -------------------------
# Time Series
# -------------------------
def make_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    agg: str = "sum",
    freq: str = "D",
    group_col: Optional[str] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Create a time series line chart with optional grouping.

    Parameters:
    - df: input DataFrame
    - date_col: column containing dates
    - value_col: numeric column to aggregate
    - agg: 'sum', 'mean', 'count', 'median'
    - freq: resample frequency ('D', 'W', 'M', 'Q', 'Y')
    - group_col: optional categorical column for multiple lines
    - title: optional custom title

    Returns:
    - plotly Figure
    """
    _ensure_column_exists(df, date_col)
    _ensure_column_exists(df, value_col)
    _ensure_column_exists(df, group_col, allow_none=True)

    # Parse dates
    ser_dt = pd.to_datetime(df[date_col], errors="coerce")
    if ser_dt.isna().all():
        raise ValueError(f"Column '{date_col}' contains no valid dates.")

    tmp = df.copy()
    tmp[date_col] = ser_dt
    tmp = tmp.dropna(subset=[date_col])  # Remove rows with invalid dates
    tmp = tmp.set_index(date_col)

    # Aggregation mapping
    agg_map = {
        "sum": "sum",
        "mean": "mean",
        "count": "count",
        "median": "median",
    }
    agg_func = agg_map.get(agg, "sum")

    # With grouping
    if group_col and group_col in tmp.columns and group_col != "":
        grouped = (
            tmp[[value_col, group_col]]
            .groupby(group_col)
            .resample(freq)[value_col]
            .agg(agg_func)
            .reset_index()
        )
        fig = px.line(
            grouped,
            x=date_col,
            y=value_col,
            color=group_col,
            title=title or f"{value_col} over time by {group_col} ({agg})",
            markers=True
        )
    else:
        # No grouping
        series = tmp[value_col].resample(freq).agg(agg_func).reset_index()
        fig = px.line(
            series,
            x=date_col,
            y=value_col,
            title=title or f"{value_col} over time ({agg})",
            markers=True
        )

    fig.update_layout(
        xaxis_title=date_col,
        yaxis_title=f"{value_col} ({agg})",
        hovermode="x unified"
    )
    return fig


# -------------------------
# Distribution (Histogram + Box Plot)
# -------------------------
def make_distribution(
    df: pd.DataFrame,
    col: str,
    nbins: int = 30,
    chart_type: str = "histogram",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Create distribution visualizations (histogram or box plot).

    Parameters:
    - df: input DataFrame
    - col: numeric column
    - nbins: number of bins for histogram
    - chart_type: 'histogram' or 'box'
    - title: optional title
    """
    _ensure_column_exists(df, col)
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ValueError(f"Column '{col}' must be numeric for distribution plot.")

    # Remove NaN values
    data = df[col].dropna()
    
    if chart_type == "box":
        fig = px.box(df, y=col, title=title or f"Box Plot of {col}", points="outliers")
        fig.update_layout(yaxis_title=col)
    else:
        fig = px.histogram(
            df,
            x=col,
            nbins=nbins,
            marginal="box",
            title=title or f"Distribution of {col}"
        )
        fig.update_layout(
            xaxis_title=col,
            yaxis_title="Count",
            bargap=0.1
        )

    return fig


# -------------------------
# Category Analysis (Bar/Pie)
# -------------------------
def make_category_chart(
    df: pd.DataFrame,
    cat_col: str,
    agg_col: Optional[str] = None,
    agg: str = "count",
    chart: str = "bar",
    top_n: int = 20,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Create bar or pie chart for categorical analysis.

    Parameters:
    - df: input DataFrame
    - cat_col: categorical column
    - agg_col: numeric column to aggregate (None for count)
    - agg: 'count', 'sum', 'mean', 'median'
    - chart: 'bar' or 'pie'
    - top_n: show only top N categories
    - title: optional
    """
    _ensure_column_exists(df, cat_col)

    # Count aggregation
    if agg == "count" or agg_col is None or agg_col == "":
        agg_series = df[cat_col].value_counts().head(top_n).reset_index()
        agg_series.columns = [cat_col, "value"]
        y_label = "Count"
    else:
        _ensure_column_exists(df, agg_col)
        if not pd.api.types.is_numeric_dtype(df[agg_col]):
            raise ValueError(f"Column '{agg_col}' must be numeric for aggregation.")
        
        # Aggregate by category
        if agg == "sum":
            temp = df.groupby(cat_col)[agg_col].sum()
        elif agg == "mean":
            temp = df.groupby(cat_col)[agg_col].mean()
        elif agg == "median":
            temp = df.groupby(cat_col)[agg_col].median()
        else:
            temp = df.groupby(cat_col)[agg_col].count()
        
        agg_series = temp.sort_values(ascending=False).head(top_n).reset_index()
        agg_series.columns = [cat_col, "value"]
        y_label = f"{agg.capitalize()} of {agg_col}"

    # Create chart
    if chart == "pie":
        fig = px.pie(
            agg_series,
            names=cat_col,
            values="value",
            title=title or f"Distribution of {cat_col}"
        )
    else:
        fig = px.bar(
            agg_series,
            x=cat_col,
            y="value",
            title=title or f"{cat_col} by {agg}",
            text="value"
        )
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig.update_layout(
            xaxis_title=cat_col,
            yaxis_title=y_label,
            xaxis={'categoryorder': 'total descending'}
        )

    return fig


# -------------------------
# Scatter Plot or Correlation Heatmap
# -------------------------
def make_scatter_or_heatmap(
    df: pd.DataFrame,
    col_x: Optional[str] = None,
    col_y: Optional[str] = None,
    chart_type: str = "scatter",
    corr_method: str = "pearson",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Create scatter plot or correlation heatmap.

    Parameters:
    - df: input DataFrame
    - col_x, col_y: columns for scatter plot
    - chart_type: 'scatter' or 'heatmap'
    - corr_method: 'pearson', 'spearman', or 'kendall'
    - title: optional
    """
    if chart_type == "scatter":
        if not col_x or not col_y:
            raise ValueError("Scatter plot requires both X and Y columns.")
        
        _ensure_column_exists(df, col_x)
        _ensure_column_exists(df, col_y)
        
        if not pd.api.types.is_numeric_dtype(df[col_x]):
            raise ValueError(f"Column '{col_x}' must be numeric for scatter plot.")
        if not pd.api.types.is_numeric_dtype(df[col_y]):
            raise ValueError(f"Column '{col_y}' must be numeric for scatter plot.")
        
        # Create scatter WITHOUT trendline (to avoid statsmodels dependency)
        fig = px.scatter(
            df,
            x=col_x,
            y=col_y,
            title=title or f"Scatter: {col_x} vs {col_y}",
            opacity=0.6
        )
        
        # Add manual trendline using numpy
        try:
            # Remove NaN values
            clean_df = df[[col_x, col_y]].dropna()
            if len(clean_df) > 1:
                # Calculate linear regression manually
                x_vals = clean_df[col_x].values
                y_vals = clean_df[col_y].values
                
                # Linear fit: y = mx + b
                coeffs = np.polyfit(x_vals, y_vals, 1)
                m, b = coeffs[0], coeffs[1]
                
                # Create trendline
                x_range = np.array([x_vals.min(), x_vals.max()])
                y_trend = m * x_range + b
                
                # Add trendline to figure
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=y_trend,
                    mode='lines',
                    name='Trendline',
                    line=dict(color='red', dash='dash')
                ))
        except Exception as e:
            print(f"Could not add trendline: {e}")
        
        fig.update_layout(
            xaxis_title=col_x,
            yaxis_title=col_y
        )
        
    else:  # heatmap
        # Get all numeric columns
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            raise ValueError("Need at least 2 numeric columns for correlation heatmap.")
        
        # Calculate correlation
        corr = num_df.corr(method=corr_method)
        
        # Create heatmap
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title=title or f"Correlation Matrix ({corr_method})",
            zmin=-1,
            zmax=1
        )
        fig.update_layout(
            xaxis_title="",
            yaxis_title=""
        )

    return fig


# -------------------------
# Export to PNG
# -------------------------
def fig_to_png_bytes(fig: go.Figure, width: int = 1200, height: int = 600) -> bytes:
    """
    Convert Plotly figure to PNG bytes.
    
    Requires: pip install -U kaleido
    
    Parameters:
    - fig: Plotly figure
    - width, height: image dimensions
    
    Returns:
    - PNG image as bytes
    """
    try:
        img_bytes = fig.to_image(format="png", width=width, height=height, engine="kaleido")
        return img_bytes
    except Exception as e:
        raise RuntimeError(
            f"Failed to export PNG. Install kaleido: pip install -U kaleido\nError: {e}"
        )