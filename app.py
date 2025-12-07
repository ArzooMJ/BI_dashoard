# app.py
import numpy as np
import gradio as gr
import pandas as pd
from utils.filtering import (
    get_column_metadata,
    get_numeric_bounds,
    get_categorical_options,
    get_date_bounds,
    get_numeric_unique_values,
    get_date_unique_values,
    apply_all_filters
)
from utils.visualizations import (
    make_time_series,
    make_distribution,
    make_category_chart,
    make_scatter_or_heatmap,
    fig_to_png_bytes,
)
from utils.insights import generate_insights
from utils.data_processor import DataProcessor
import io
import json
import os

dp = DataProcessor()

def _df_or_none():
    return dp.get_dataframe()

def load_file(upload):
    try:
        df = dp.load_file(upload)
        dp.set_dataframe(df)
        dp.set_filtered(df.copy())
        return f"Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns", df.head(10)
    except Exception as e:
        return f"Error: {str(e)}", pd.DataFrame()
    
def load_sample(path):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Sample file not found: {path}")
        # read with pandas directly to ensure proper DataFrame
        df = pd.read_csv(path, low_memory=False)
        dp.set_dataframe(df)
        dp.set_filtered(df.copy())
        return f"Sample Loaded: {df.shape[0]} rows, {df.shape[1]} columns", df.head(10)
    except Exception as e:
        return f"Error loading sample: {str(e)}", pd.DataFrame()


# -------------------------
# 2. Statistics Tab
# -------------------------
def get_statistics():
    if dp.get_dataframe() is None:
        return "No data loaded.", pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    desc, cat_df = dp.summary_statistics()
    miss = dp.missing_report()
    corr = dp.correlation_matrix()
    return "Statistics computed.", desc, cat_df, miss, corr

# -------------------------
# 3. Filtering Tab
# -------------------------
def inspect_dataset_for_ui():
    df=dp.get_dataframe()
    if df is None:
        return (
            gr.update(value="No data loaded"),
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=[]),
        )
    
    meta = get_column_metadata(df)

    summary={
        "num": meta.get("numeric",[]),
        "cat": meta.get("categorical", []),
        "date": meta.get("date", [])
    }
    summary_text=json.dumps({k: (len(v) if isinstance(v,list) else v) for k, v in summary.items()}, indent=2)

    num_choices=meta.get("numeric", [])
    cat_choices=meta.get("categorical", [])
    date_choices=meta.get("date", [])

    return (
        gr.update(value=summary_text),
        gr.update(choices=num_choices, value=None),
        gr.update(choices=cat_choices, value=None),
        gr.update(choices=date_choices, value=None),
    )

def update_numeric_controls(col_name):
    df = dp.get_dataframe()
    # If no df or no column selected, hide numeric controls
    if df is None or not col_name:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="")
        )
    try:
        col_min, col_max, step = get_numeric_bounds(df, col_name)
        # build updates for Gradio components already present in your UI
        slider_update = gr.update(value=(col_min, col_max), minimum=col_min, maximum=col_max, step=step, visible=True)
        min_update = gr.update(value=col_min, visible=True)
        max_update = gr.update(value=col_max, visible=True)
        label_update = gr.update(value=f"{col_name} range: {col_min} — {col_max}", visible=True)
        return slider_update, min_update, max_update, label_update
    except Exception as e:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=str(e))
        )
    
def update_categorical_choices(col_name):
    df = dp.get_dataframe()
    if df is None or not col_name:
        # CheckboxGroup expects choices and visible flag in your UI (we return two values)
        return gr.update(choices=[], value=[]), gr.update(visible=False)
    try:
        options = get_categorical_options(df, col_name, max_options=500)
        # return choices update and visible update for the CheckboxGroup
        return gr.update(choices=options, value=[]), gr.update(visible=True)
    except Exception as e:
        return gr.update(choices=[] , value=[]), gr.update(visible=False)


def update_date_bounds(col_name):
    df = dp.get_dataframe()
    if df is None or not col_name:
        return gr.update(visible=False), gr.update(visible=False)
    try:
        start, end = get_date_bounds(df, col_name)
        if start is None or end is None:
            return gr.update(visible=False), gr.update(visible=False)
        return gr.update(value=start, visible=True), gr.update(value=end, visible=True)
    except Exception as e:
        return gr.update(visible=False), gr.update(visible=False)


def apply_filters_from_state(state_filters):
    df = dp.get_dataframe()
    if df is None:
        return "No data loaded.", pd.DataFrame(), 0
    try:
        filtered_df, row_count = apply_all_filters(
            df,
            state_filters.get("numerical", {}),
            state_filters.get("categorical", {}),
            state_filters.get("date", {})
        )
        dp.set_filtered(filtered_df)
        return f"Filtered rows: {row_count}", filtered_df.head(10), row_count
    except Exception as e:
        return f"Error applying filters: {str(e)}", pd.DataFrame(), 0


def apply_filters_ui(filters_json):
    df = dp.get_dataframe()
    if df is None:
        return "No data loaded", pd.DataFrame()
    
    try:
        filters = json.loads(filters_json) if filters_json else {}
        num_filters = filters.get("numerical", {})
        cat_filters = filters.get("categorical", {})
        date_filters = filters.get("date", {})
        filtered_df, row_count = apply_all_filters(
            df, num_filters, cat_filters, date_filters
        )
        dp.set_filtered(filtered_df)
        return f"Filtered rows: {row_count}", filtered_df.head(10)
    except Exception as e:
        return f"Error applying filters: {str(e)}", pd.DataFrame()

# -------------------------
# 4. Visualizations Tab
# -------------------------
def create_visualization(viz_type, col_x, col_y, agg, group_col, chart_type, freq):
    df_filtered = dp.get_filtered()
    if df_filtered is None:
        return None
    try:
        if viz_type == "Time Series":
            fig = make_time_series(
                df_filtered, date_col=col_x, value_col=col_y, agg=agg, freq=freq, group_col=group_col
            )
        elif viz_type == "Distribution":
            fig = make_distribution(df_filtered, col=col_x)
        elif viz_type == "Category":
            fig = make_category_chart(
                df_filtered, cat_col=col_x, agg_col=col_y, agg=agg, chart=chart_type
            )
        elif viz_type == "Scatter/Heatmap":
            fig = make_scatter_or_heatmap(df_filtered, col_x=col_x, col_y=col_y)
        else:
            fig = None
    except Exception as e:
        print("Visualization error:", e)
        fig = None
    return fig

def export_visualization(fig):
    if fig is None:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

def generate_viz(df, viz_type,
                 date_col=None, value_col=None, group_col=None,
                 col=None, agg_col=None, agg="count", chart="bar",
                 col_x=None, col_y=None, corr_method="pearson"):
    """
    Unified visualization generator for Gradio.

    Parameters depend on the chart type:
    - Time Series: date_col, value_col, group_col, agg
    - Distribution: col
    - Category: cat_col (as col), agg_col, agg, chart
    - Scatter/Heatmap: col_x, col_y, corr_method
    """
    # Build kwargs dynamically
    kwargs = {}

    if viz_type.lower() == "time series":
        kwargs = {
            "date_col": date_col,
            "value_col": value_col,
            "group_col": group_col,
            "agg": agg
        }
    elif viz_type.lower() == "distribution":
        kwargs = {"col": col}
    elif viz_type.lower() == "category":
        kwargs = {"cat_col": col, "agg_col": agg_col, "agg": agg, "chart": chart}
    elif viz_type.lower() == "scatter/heatmap":
        kwargs = {"col_x": col_x, "col_y": col_y, "corr_method": corr_method}

    try:
        fig = create_visualization(df, viz_type, **kwargs)
        return fig
    except Exception as e:
        return f"Error generating visualization: {e}"

# -------------------------
# 5. Insights Tab
# -------------------------
def get_insights(date_col=None):
    df_filtered = dp.get_filtered()
    if df_filtered is None:
        return "No data loaded.", {}
    numeric_cols = dp.numeric_columns()
    insights = generate_insights(df_filtered, numeric_cols=numeric_cols, date_col=date_col)
    return "Insights generated.", insights

# -------------------------
# 6. Export
# -------------------------
def export_filtered_data():
    """Export filtered data to CSV file."""
    df_filtered = dp.get_filtered()
    
    if df_filtered is None or df_filtered.empty:
        return None, " No data to export. Please load and filter data first."
    
    try:
        import tempfile
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.csv',
            prefix=f'filtered_data_{timestamp}_'
        )
        
        df_filtered.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        status_msg = f"✅ Successfully exported {len(df_filtered):,} rows and {len(df_filtered.columns)} columns!"
        
        return temp_file.name, status_msg
    
    except Exception as e:
        import traceback
        print("=" * 60)
        print("EXPORT ERROR:")
        print("=" * 60)
        print(traceback.format_exc())
        print("=" * 60)
        return None, f"❌ Export failed: {str(e)}"


def export_visualization(fig):
    if fig is None:
        return None
    img_bytes = fig_to_png_bytes(fig)
    return io.BytesIO(img_bytes)

# -------------------------
# Build Gradio Interface
# -------------------------
def create_dashboard():
    with gr.Blocks() as demo:
        gr.Markdown("# Business Intelligence Dashboard")

        # --- Data Upload ---
        with gr.Tab("Data Upload"):
            file_input = gr.File(label="Upload CSV/Excel file")
            upload_btn = gr.Button("Load Data")

            sample1_btn= gr.Button("Load sample dataset 1")
            sample2_btn = gr.Button("Load Sample Dataset 2")

            upload_status = gr.Textbox()
            data_preview = gr.Dataframe(interactive=False)
            upload_btn.click(load_file, inputs=file_input, outputs=[upload_status, data_preview])

            sample1_btn.click(lambda: load_sample("data/sample_sales.csv"), inputs=None, outputs=[upload_status, data_preview])
            sample2_btn.click(lambda: load_sample("data/sample_customers.csv"), inputs=None, outputs=[upload_status, data_preview])

        with gr.Tab("Dataset Info"):
            gr.Markdown("### 📋 Dataset Information")
            gr.Markdown("Explore your dataset's structure, columns, and data types at a glance.")

            with gr.Row():
                info_btn = gr.Button("📊 Show Dataset Info", variant="primary", scale=2)
                refresh_info_btn = gr.Button("🔄 Refresh", scale=1)

            info_status = gr.Textbox(label="Status", interactive=False)

            # Main content in tabs
            with gr.Tabs():
                # Tab 1: Quick Overview
                with gr.Tab("📊 Quick Overview"):
                    gr.Markdown("""
                    ### At a Glance
                    Essential information about your dataset's size and structure.
                    """)

                    with gr.Row():
                        with gr.Column(scale=1):
                            overview_md = gr.Markdown(value="*Click 'Show Dataset Info' to load overview*")
                        with gr.Column(scale=1):
                            memory_md = gr.Markdown(value="")

                # Tab 2: Column Details
                with gr.Tab("📑 Column Details"):
                    gr.Markdown("""
                    ### Column Information
                    Detailed breakdown of each column including name, type, and characteristics.
                    """)

                    column_details_df = gr.Dataframe(
                        label="Column Information",
                        interactive=False
                    )

                # Tab 3: Data Types Summary
                with gr.Tab("🏷️ Data Types"):
                    gr.Markdown("""
                    ### Data Type Distribution
                    Understand what types of data you're working with.
                    """)

                    with gr.Row():
                        with gr.Column():
                            dtype_summary_md = gr.Markdown(value="")
                        with gr.Column():
                            dtype_chart_md = gr.Markdown(value="")

                # Tab 4: Sample Data
                with gr.Tab("👀 Preview"):
                    gr.Markdown("""
                    ### Data Preview
                    See the first and last rows of your dataset.
                    """)

                    with gr.Accordion("🔝 First 10 Rows", open=True):
                        preview_head_df = gr.Dataframe(
                            label="Top Rows",
                            interactive=False
                        )

                    with gr.Accordion("🔻 Last 10 Rows", open=False):
                        preview_tail_df = gr.Dataframe(
                            label="Bottom Rows",
                            interactive=False
                        )

            # ===== Helper Functions =====

            def show_enhanced_dataset_info():
                """Show comprehensive dataset information."""
                df = dp.get_dataframe()

                if df is None:
                    return (
                        "❌ No data loaded. Please upload data first.",
                        "*No data available*",
                        "",
                        pd.DataFrame(),
                        "",
                        "",
                        pd.DataFrame(),
                        pd.DataFrame()
                    )

                try:
                    # 1. Quick Overview
                    total_rows = len(df)
                    total_cols = len(df.columns)

                    # Calculate memory usage
                    memory_bytes = df.memory_usage(deep=True).sum()
                    memory_mb = memory_bytes / (1024 ** 2)
                    memory_kb = memory_bytes / 1024

                    if memory_mb > 1:
                        memory_str = f"{memory_mb:.2f} MB"
                    else:
                        memory_str = f"{memory_kb:.2f} KB"

                    # Count column types
                    numeric_count = len(df.select_dtypes(include=[np.number]).columns)
                    text_count = len(df.select_dtypes(include=['object']).columns)
                    datetime_count = len(df.select_dtypes(include=['datetime64']).columns)
                    bool_count = len(df.select_dtypes(include=['bool']).columns)
                    category_count = len(df.select_dtypes(include=['category']).columns)

                    overview = f"""
        ### 📊 Dataset Size
        - **Total Rows**: {total_rows:,}
        - **Total Columns**: {total_cols}
        - **Total Cells**: {(total_rows * total_cols):,}

        ### 📁 Column Types
        - **Numeric**: {numeric_count} column(s)
        - **Text/Object**: {text_count} column(s)
        - **DateTime**: {datetime_count} column(s)
        - **Boolean**: {bool_count} column(s)
        - **Category**: {category_count} column(s)
        """

                    memory_info = f"""
        ### 💾 Memory Usage
        - **Total Size**: {memory_str}
        - **Per Row**: {(memory_bytes / total_rows / 1024):.2f} KB
        - **Per Column**: {(memory_bytes / total_cols / 1024):.2f} KB

        ### 🔢 Data Density
        - **Missing Cells**: {df.isnull().sum().sum():,}
        - **Filled Cells**: {(total_rows * total_cols - df.isnull().sum().sum()):,}
        - **Fill Rate**: {((1 - df.isnull().sum().sum() / (total_rows * total_cols)) * 100):.1f}%
        """

                    # 2. Column Details
                    column_info = []
                    for col in df.columns:
                        dtype = str(df[col].dtype)
                        non_null = df[col].count()
                        null_count = df[col].isnull().sum()
                        null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0

                        # Determine column category
                        if pd.api.types.is_numeric_dtype(df[col]):
                            col_category = "Numeric"
                            unique_vals = df[col].nunique()
                            sample_val = str(df[col].dropna().iloc[0]) if non_null > 0 else "N/A"
                        elif pd.api.types.is_datetime64_any_dtype(df[col]):
                            col_category = "DateTime"
                            unique_vals = df[col].nunique()
                            sample_val = str(df[col].dropna().iloc[0]) if non_null > 0 else "N/A"
                        else:
                            col_category = "Text/Object"
                            unique_vals = df[col].nunique()
                            sample_val = str(df[col].dropna().iloc[0])[:50] if non_null > 0 else "N/A"

                        column_info.append({
                            'Column Name': col,
                            'Data Type': dtype,
                            'Category': col_category,
                            'Non-Null': f"{non_null:,}",
                            'Null Count': f"{null_count:,}",
                            'Null %': f"{null_pct:.1f}%",
                            'Unique Values': f"{unique_vals:,}",
                            'Sample Value': sample_val
                        })

                    column_details = pd.DataFrame(column_info)

                    # 3. Data Types Summary
                    dtype_counts = df.dtypes.value_counts()

                    dtype_summary = "### 🏷️ Data Type Breakdown\n\n"
                    for dtype, count in dtype_counts.items():
                        dtype_summary += f"- **{dtype}**: {count} column(s)\n"

                    # Create a simple text chart
                    dtype_chart = "### 📊 Visual Distribution\n\n```\n"
                    max_count = dtype_counts.max()
                    for dtype, count in dtype_counts.items():
                        bar_length = int((count / max_count) * 30)
                        bar = "█" * bar_length
                        dtype_chart += f"{str(dtype):<15} {bar} {count}\n"
                    dtype_chart += "```"

                    # 4. Preview Data
                    preview_head = df.head(10)
                    preview_tail = df.tail(10)

                    status_msg = f"✅ Dataset info loaded successfully! {total_rows:,} rows × {total_cols} columns"

                    return (
                        status_msg,
                        overview,
                        memory_info,
                        column_details,
                        dtype_summary,
                        dtype_chart,
                        preview_head,
                        preview_tail
                    )

                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    print("=" * 60)
                    print("DATASET INFO ERROR:")
                    print("=" * 60)
                    print(error_details)
                    print("=" * 60)

                    return (
                        f"❌ Error loading dataset info: {str(e)}",
                        "*Error*", "", pd.DataFrame(), "", "", pd.DataFrame(), pd.DataFrame()
                    )

            # ===== Wire Up Events =====

            info_btn.click(
                show_enhanced_dataset_info,
                inputs=[],
                outputs=[
                    info_status,
                    overview_md,
                    memory_md,
                    column_details_df,
                    dtype_summary_md,
                    dtype_chart_md,
                    preview_head_df,
                    preview_tail_df
                ]
            )

            refresh_info_btn.click(
                show_enhanced_dataset_info,
                inputs=[],
                outputs=[
                    info_status,
                    overview_md,
                    memory_md,
                    column_details_df,
                    dtype_summary_md,
                    dtype_chart_md,
                    preview_head_df,
                    preview_tail_df
                ]
            )

        # --- Statistics ---
        with gr.Tab("Statistics"):
            gr.Markdown("### 📊 Statistical Analysis")
            gr.Markdown("Comprehensive statistical overview of your dataset's numeric and categorical columns.")

            with gr.Row():
                stats_btn = gr.Button("📈 Compute Statistics", variant="primary", scale=2)
                refresh_stats_btn = gr.Button("🔄 Refresh", scale=1)

            stats_status = gr.Textbox(label="Status", interactive=False)

            # Tabbed statistics display
            with gr.Tabs():
                # Tab 1: Overview
                with gr.Tab("📋 Overview"):
                    gr.Markdown("""
                    ### Dataset Summary
                    High-level overview of your data's structure and composition.
                    """)

                    with gr.Row():
                        with gr.Column():
                            dataset_info_md = gr.Markdown(value="*Compute statistics to see overview*")
                        with gr.Column():
                            data_quality_md = gr.Markdown(value="")

                # Tab 2: Numeric Statistics
                with gr.Tab("🔢 Numeric Columns"):
                    gr.Markdown("""
                    ### Descriptive Statistics for Numeric Data
                    Understand the central tendency, spread, and distribution of your numeric columns.

                    **Key Metrics:**
                    - **count**: Number of non-null values
                    - **mean**: Average value
                    - **std**: Standard deviation (measure of spread)
                    - **min**: Minimum value
                    - **25%**: First quartile (25th percentile)
                    - **50%**: Median (50th percentile)
                    - **75%**: Third quartile (75th percentile)
                    - **max**: Maximum value
                    """)

                    numeric_stats_df = gr.Dataframe(
                        label="Numeric Column Statistics",
                        interactive=False
                    )

                # Tab 3: Categorical Statistics
                with gr.Tab("📑 Categorical Columns"):
                    gr.Markdown("""
                    ### Categorical Data Analysis
                    Analyze the diversity and distribution of your text/categorical columns.

                    **Key Metrics:**
                    - **unique_values**: Number of distinct values
                    - **most_frequent**: Most common value in the column
                    """)

                    categorical_stats_df = gr.Dataframe(
                        label="Categorical Column Statistics",
                        interactive=False
                    )

                # Tab 4: Missing Values
                with gr.Tab("⚠️ Missing Values"):
                    gr.Markdown("""
                    ### Data Completeness Analysis
                    Identify columns with missing data that may need attention.

                    **Why it matters:**
                    - Missing data can affect analysis accuracy
                    - High percentages (>20%) may require imputation or removal
                    - Patterns in missing data can reveal data quality issues
                    """)

                    missing_summary_md = gr.Markdown(value="")
                    missing_report_df = gr.Dataframe(
                        label="Missing Values by Column",
                        interactive=False
                    )

                # Tab 5: Correlations
                with gr.Tab("🔗 Correlation Matrix"):
                    gr.Markdown("""
                    ### Numeric Column Correlations
                    Shows relationships between numeric variables (-1 to +1).

                    **Interpretation:**
                    - **+1**: Perfect positive correlation (variables move together)
                    - **0**: No correlation (variables are independent)
                    - **-1**: Perfect negative correlation (variables move oppositely)

                    **Strength Guidelines:**
                    - **0.7-1.0 or -0.7 to -1.0**: Strong correlation
                    - **0.3-0.7 or -0.3 to -0.7**: Moderate correlation
                    - **0-0.3 or 0 to -0.3**: Weak correlation
                    """)

                    correlation_matrix_df = gr.Dataframe(
                        label="Correlation Matrix",
                        interactive=False
                    )

            # ===== Helper Functions =====

            def compute_enhanced_statistics():
                """Compute comprehensive statistics using DataProcessor methods."""
                df = dp.get_dataframe()

                if df is None:
                    return (
                        "❌ No data loaded. Please upload data first.",
                        "",  # dataset_info_md
                        "",  # data_quality_md
                        pd.DataFrame(),  # numeric_stats
                        pd.DataFrame(),  # categorical_stats
                        "",  # missing_summary_md
                        pd.DataFrame(),  # missing_report
                        pd.DataFrame()   # correlation_matrix
                    )

                try:
                    # 1. Dataset Overview
                    total_rows = len(df)
                    total_cols = len(df.columns)
                    numeric_cols = dp.numeric_columns()
                    categorical_cols = dp.categorical_columns()
                    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

                    dataset_info = f"""
        ### 📊 Dataset Structure
        - **Total Rows**: {total_rows:,}
        - **Total Columns**: {total_cols}
        - **Numeric Columns**: {len(numeric_cols)}
        - **Categorical Columns**: {len(categorical_cols)}
        - **Date Columns**: {len(datetime_cols)}
        - **Memory Usage**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
        """

                    # 2. Data Quality Overview
                    total_cells = total_rows * total_cols
                    missing_cells = df.isnull().sum().sum()
                    missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0

                    duplicate_rows = df.duplicated().sum()

                    quality_icon = "✅" if missing_pct < 5 else "⚠️" if missing_pct < 20 else "❌"

                    data_quality = f"""
        ### {quality_icon} Data Quality
        - **Missing Values**: {missing_cells:,} ({missing_pct:.2f}%)
        - **Complete Rows**: {df.dropna().shape[0]:,} ({(df.dropna().shape[0]/total_rows*100):.1f}%)
        - **Duplicate Rows**: {duplicate_rows:,}
        - **Data Quality Score**: {'Excellent' if missing_pct < 5 else 'Good' if missing_pct < 20 else 'Needs Attention'}
        """

                    # 3. Numeric Statistics (using DataProcessor method)
                    numeric_desc, cat_summary = dp.summary_statistics()

                    if numeric_desc is not None and not numeric_desc.empty:
                        # Add column name as first column for better readability
                        numeric_stats = numeric_desc.round(2)
                        numeric_stats.insert(0, 'Column', numeric_stats.index)
                        numeric_stats = numeric_stats.reset_index(drop=True)
                    else:
                        numeric_stats = pd.DataFrame({"Message": ["No numeric columns found"]})

                    # 4. Categorical Statistics (using DataProcessor method)
                    if cat_summary is not None and not cat_summary.empty:
                        categorical_stats = cat_summary.copy()
                        categorical_stats.insert(0, 'Column', categorical_stats.index)
                        categorical_stats = categorical_stats.reset_index(drop=True)
                        # Truncate long values
                        if 'most_frequent' in categorical_stats.columns:
                            categorical_stats['most_frequent'] = categorical_stats['most_frequent'].astype(str).str[:50]
                    else:
                        categorical_stats = pd.DataFrame({"Message": ["No categorical columns found"]})

                    # 5. Missing Values Analysis (using DataProcessor method)
                    missing_data = dp.missing_report()

                    if missing_data is not None and not missing_data.empty:
                        # Filter only columns with missing values
                        missing_data_filtered = missing_data[missing_data['missing_count'] > 0].copy()

                        if len(missing_data_filtered) > 0:
                            # Sort by missing count
                            missing_data_filtered = missing_data_filtered.sort_values('missing_count', ascending=False)

                            missing_summary = f"""
        ### ⚠️ Missing Data Summary
        - **Columns with missing data**: {len(missing_data_filtered)}
        - **Total missing values**: {int(missing_data_filtered['missing_count'].sum()):,}
        - **Worst column**: {missing_data_filtered.index[0]} ({int(missing_data_filtered['missing_count'].iloc[0]):,} missing, {missing_data_filtered['missing_percent'].iloc[0]:.1f}%)

        **Recommendation**: Consider imputation or removal for columns with >20% missing data.
        """

                            # Prepare display DataFrame
                            missing_report = missing_data_filtered.copy()
                            missing_report.insert(0, 'Column', missing_report.index)
                            missing_report = missing_report.reset_index(drop=True)
                            missing_report['missing_count'] = missing_report['missing_count'].astype(int)
                            missing_report['missing_percent'] = missing_report['missing_percent'].round(2).astype(str) + '%'

                            # Add data type
                            missing_report['Data Type'] = [str(df[col].dtype) for col in missing_data_filtered.index]
                        else:
                            missing_summary = "### ✅ No Missing Data\n\nGreat! Your dataset has no missing values."
                            missing_report = pd.DataFrame({"Message": ["No missing values detected"]})
                    else:
                        missing_summary = "### ✅ No Missing Data\n\nGreat! Your dataset has no missing values."
                        missing_report = pd.DataFrame({"Message": ["No missing values detected"]})

                    # 6. Correlation Matrix (using DataProcessor method)
                    corr_matrix = dp.correlation_matrix()

                    if corr_matrix is not None and not corr_matrix.empty:
                        # Format for better display
                        corr_display = corr_matrix.round(3)
                        corr_display.insert(0, 'Column', corr_display.index)
                        corr_display = corr_display.reset_index(drop=True)
                    else:
                        corr_display = pd.DataFrame({"Message": ["Need at least 2 numeric columns for correlation"]})

                    status_msg = f"✅ Statistics computed successfully! Analyzed {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns."

                    return (
                        status_msg,
                        dataset_info,
                        data_quality,
                        numeric_stats,
                        categorical_stats,
                        missing_summary,
                        missing_report,
                        corr_display
                    )

                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    print("=" * 60)
                    print("STATISTICS ERROR:")
                    print("=" * 60)
                    print(error_details)
                    print("=" * 60)

                    return (
                        f"❌ Error computing statistics: {str(e)}",
                        "", "", pd.DataFrame(), pd.DataFrame(), "", pd.DataFrame(), pd.DataFrame()
                    )

            # ===== Wire Up Events =====

            stats_btn.click(
                compute_enhanced_statistics,
                inputs=[],
                outputs=[
                    stats_status,
                    dataset_info_md,
                    data_quality_md,
                    numeric_stats_df,
                    categorical_stats_df,
                    missing_summary_md,
                    missing_report_df,
                    correlation_matrix_df
                ]
            )

            refresh_stats_btn.click(
                compute_enhanced_statistics,
                inputs=[],
                outputs=[
                    stats_status,
                    dataset_info_md,
                    data_quality_md,
                    numeric_stats_df,
                    categorical_stats_df,
                    missing_summary_md,
                    missing_report_df,
                    correlation_matrix_df
                ]
            )

            # filter tab
        with gr.Tab("Filter & Explore"):
            gr.Markdown("### 🔍 Interactive Filters")

            # --- Generate metadata ---
            gen_meta_btn = gr.Button("Inspect dataset and generate filter controls")
            meta_output = gr.Textbox(label="Detected column metadata (summary)")

            # --- Numeric controls (min/max dropdowns) ---
            gr.Markdown("#### Numeric column")
            num_col_select = gr.Dropdown(label="Select numeric column", choices=[], interactive=True)
            num_range_label = gr.Textbox(label="Column range", interactive=False, visible=False)
            num_min_dropdown = gr.Dropdown(label="Min value (choose)", choices=[], value=None, visible=False, interactive=True)
            num_max_dropdown = gr.Dropdown(label="Max value (choose)", choices=[], value=None, visible=False, interactive=True)
            add_num_btn = gr.Button("Add Numeric Filter")

            # --- Categorical controls ---
            gr.Markdown("#### Categorical column")
            cat_col_select = gr.Dropdown(label="Select categorical column", choices=[], interactive=True)
            cat_multiselect = gr.CheckboxGroup(choices=[], label="Select values", visible=False)
            add_cat_btn = gr.Button("Add Categorical Filter")

            # --- Date controls (min/max dropdowns) ---
            gr.Markdown("#### Date column")
            date_col_select = gr.Dropdown(label="Select date column", choices=[], interactive=True)
            date_range_label = gr.Textbox(label="Date range", interactive=False, visible=False)
            date_min_dropdown = gr.Dropdown(label="Start date (choose)", choices=[], value=None, visible=False, interactive=True)
            date_max_dropdown = gr.Dropdown(label="End date (choose)", choices=[], value=None, visible=False, interactive=True)
            add_date_btn = gr.Button("Add Date Filter")

            # --- Columns to display ---
            display_cols = gr.CheckboxGroup(choices=[], label="Columns to display (select none = show all)", visible=True)

            # --- Filters state & outputs ---
            filters_state = gr.State({"numerical": {}, "categorical": {}, "date": {}})
            filters_display = gr.JSON(label="Current Filters")
            apply_filters_btn = gr.Button("Apply Filters")
            apply_status = gr.Textbox(label="Filter Status")
            filter_preview = gr.Dataframe(headers=None, interactive=False)
            row_count_box = gr.Number(label="Rows after filtering")

            # --- Helper callbacks ---
            def inspect_dataset_for_ui_cb():
                """Load filter controls with readable metadata."""
                df = dp.get_dataframe()
                if df is None:
                    return (
                        gr.update(value="❌ No data loaded"),
                        gr.update(choices=[], value=None),
                        gr.update(choices=[], value=None),
                        gr.update(choices=[], value=None),
                    )

                meta = get_column_metadata(df)

                # Format as readable text instead of JSON
                summary_text = f"""📊 Column Type Summary:

            ✓ Numeric Columns: {len(meta.get("numeric", []))}
            ✓ Categorical Columns: {len(meta.get("categorical", []))}  
            ✓ Date Columns: {len(meta.get("date", []))}

            Total: {len(meta.get("numeric", [])) + len(meta.get("categorical", [])) + len(meta.get("date", []))} columns detected
            """

                return (
                    gr.update(value=summary_text),
                    gr.update(choices=meta.get("numeric", []), value=None),
                    gr.update(choices=meta.get("categorical", []), value=None),
                    gr.update(choices=meta.get("date", []), value=None),
                )
            

            def populate_display_cols_cb():
                """Populate display columns."""
                df = dp.get_dataframe()
                if df is None:
                    return gr.update(choices=[])
                cols = df.columns.tolist()
                return gr.update(choices=cols)

            # --- FIXED: When a numeric column is chosen, populate min/max dropdowns ---
            def on_num_col_change_cb(col_name):
                """Populate numeric dropdowns - FIXED VERSION."""
                print(f"\n{'='*60}")
                print(f"DEBUG: on_num_col_change_cb called with: '{col_name}'")
                print(f"{'='*60}")

                df = dp.get_dataframe()

                if df is None or not col_name:
                    print("DEBUG: Returning invisible dropdowns (no df or col)")
                    return (
                        gr.update(visible=False, choices=[], value=None),
                        gr.update(visible=False, choices=[], value=None),
                        gr.update(value="", visible=False)
                    )

                try:
                    print(f"DEBUG: Getting bounds for column '{col_name}'")

                    # Get min, max, step using your existing function
                    col_min, col_max, _ = get_numeric_bounds(df, col_name)

                    if col_min is None:
                        print(f"DEBUG: No numeric data in {col_name}")
                        return (
                            gr.update(visible=False, choices=[], value=None),
                            gr.update(visible=False, choices=[], value=None),
                            gr.update(value=f"No numeric data in {col_name}", visible=True)
                        )

                    print(f"DEBUG: Column range: {col_min} to {col_max}")

                    label = f"{col_name} range: {col_min} — {col_max}"

                    # Get unique values using your existing function - LIMIT TO 50 for performance
                    values = get_numeric_unique_values(df, col_name, max_values=50)

                    print(f"DEBUG: Got {len(values)} unique values")

                    # Convert to strings
                    str_vals = [str(v) for v in sorted(values)]

                    print(f"DEBUG: First 3 values: {str_vals[:3]}")
                    print(f"DEBUG: Last 3 values: {str_vals[-3:]}")
                    print(f"DEBUG: Returning VISIBLE dropdowns with {len(str_vals)} choices")

                    # CRITICAL: Return with visible=True and default values selected
                    return (
                        gr.update(choices=str_vals, value=str_vals[0] if str_vals else None, visible=True, interactive=True),
                        gr.update(choices=str_vals, value=str_vals[-1] if str_vals else None, visible=True, interactive=True),
                        gr.update(value=label, visible=True)
                    )

                except Exception as e:
                    print(f"DEBUG: EXCEPTION in on_num_col_change_cb:")
                    import traceback
                    traceback.print_exc()
                    return (
                        gr.update(visible=False, choices=[], value=None),
                        gr.update(visible=False, choices=[], value=None),
                        gr.update(value=f"Error: {str(e)}", visible=True)
                    )

            # Wire up numeric column selection
            num_col_select.change(
                on_num_col_change_cb,
                inputs=[num_col_select],
                outputs=[num_min_dropdown, num_max_dropdown, num_range_label]
            )

            # --- When categorical column chosen populate checkbox choices ---
            def on_cat_change_cb(col_name):
                """Populate categorical checkboxes."""
                df = dp.get_dataframe()
                if df is None or not col_name:
                    return gr.update(choices=[], value=[]), gr.update(visible=False)

                try:
                    # Use your existing function but limit to 50 for performance
                    options = get_categorical_options(df, col_name, max_options=50)
                    return gr.update(choices=options, value=[]), gr.update(visible=True)
                except Exception as e:
                    print(f"Error in on_cat_change_cb: {e}")
                    return gr.update(choices=[], value=[]), gr.update(visible=False)

            cat_col_select.change(
                on_cat_change_cb, 
                inputs=[cat_col_select], 
                outputs=[cat_multiselect, cat_multiselect]
            )

            # --- When date column chosen populate date dropdowns ---
            def on_date_change_cb(col_name):
                """Populate date dropdowns - FIXED to show dropdowns properly."""
                print(f"\nDEBUG: Date column selected: '{col_name}'")

                df = dp.get_dataframe()

                if df is None or not col_name:
                    print("DEBUG: No df or column, hiding dropdowns")
                    return (
                        gr.update(value="", visible=False),
                        gr.update(choices=[], value=None, visible=False),
                        gr.update(choices=[], value=None, visible=False),
                    )

                try:
                    print(f"DEBUG: Processing date column '{col_name}'")

                    # Extract raw column
                    raw_series = df[col_name].astype(str).str.strip()

                    # Parse dates
                    parsed = pd.to_datetime(
                        raw_series,
                        errors="coerce",
                        infer_datetime_format=True,
                        dayfirst=False
                    )

                    # Drop NaT (invalid dates)
                    parsed = parsed.dropna()

                    if parsed.empty:
                        print("DEBUG: No valid dates found")
                        return (
                            gr.update(value="⚠️ No valid dates detected in this column.", visible=True),
                            gr.update(choices=[], value=None, visible=False),
                            gr.update(choices=[], value=None, visible=False),
                        )

                    # Sort unique date strings
                    all_dates = sorted(parsed.dt.strftime("%Y-%m-%d").unique())

                    print(f"DEBUG: Found {len(all_dates)} unique dates")

                    # Limit to 50 dates for performance
                    if len(all_dates) > 50:
                        # Sample evenly across the range
                        indices = np.linspace(0, len(all_dates)-1, 50, dtype=int)
                        date_strings = [all_dates[i] for i in indices]
                        print(f"DEBUG: Sampled to 50 dates from {len(all_dates)}")
                    else:
                        date_strings = all_dates
                        print(f"DEBUG: Using all {len(date_strings)} dates")

                    start = date_strings[0]
                    end = date_strings[-1]

                    label = f"{col_name} range: {start} — {end}"

                    print(f"DEBUG: Date range: {start} to {end}")
                    print(f"DEBUG: Returning VISIBLE date dropdowns with {len(date_strings)} choices")
                    print(f"DEBUG: Start default: {start}, End default: {end}")

                    # CRITICAL FIX: Return with visible=True, interactive=True, and default values
                    return (
                        gr.update(value=label, visible=True),
                        gr.update(choices=date_strings, value=start, visible=True, interactive=True),  # ← Fixed: added value=start
                        gr.update(choices=date_strings, value=end, visible=True, interactive=True),    # ← Fixed: added value=end
                    )

                except Exception as e:
                    print(f"DEBUG: Exception in on_date_change_cb: {e}")
                    import traceback
                    traceback.print_exc()
                    return (
                        gr.update(value=f"⚠️ Error: {str(e)}", visible=True),
                        gr.update(choices=[], value=None, visible=False),
                        gr.update(choices=[], value=None, visible=False),
                    )

            def on_date_change_cb(col_name):
                """Populate date dropdowns."""
                df = dp.get_dataframe()
                if df is None or not col_name:
                    return (
                        gr.update(value="", visible=False),
                        gr.update(choices=[], value=None, visible=False),
                        gr.update(choices=[], value=None, visible=False),
                    )

                try:
                    # Extract raw column
                    raw_series = df[col_name].astype(str).str.strip()

                    # Parse dates
                    parsed = pd.to_datetime(
                        raw_series,
                        errors="coerce",
                        infer_datetime_format=True,
                        dayfirst=False
                    )

                    # Drop NaT (invalid dates)
                    parsed = parsed.dropna()

                    if parsed.empty:
                        return (
                            gr.update(value="No valid dates detected in this column.", visible=True),
                            gr.update(choices=[], value=None, visible=False),
                            gr.update(choices=[], value=None, visible=False),
                        )

                    # Sort unique date strings - LIMIT TO 50 for performance
                    all_dates = sorted(parsed.dt.strftime("%Y-%m-%d").unique())

                    # If too many dates, sample evenly
                    if len(all_dates) > 50:
                        indices = np.linspace(0, len(all_dates)-1, 50, dtype=int)
                        date_strings = [all_dates[i] for i in indices]
                    else:
                        date_strings = all_dates

                    start = date_strings[0]
                    end = date_strings[-1]

                    label = f"{col_name} range: {start} — {end}"

                    return (
                        gr.update(value=label, visible=True),
                        gr.update(choices=date_strings, value=start, visible=True, interactive=True),
                        gr.update(choices=date_strings, value=end, visible=True, interactive=True),
                    )

                except Exception as e:
                    print(f"Error in on_date_change_cb: {e}")
                    return (
                        gr.update(value=f"Error: {str(e)}", visible=True),
                        gr.update(choices=[], value=None, visible=False),
                        gr.update(choices=[], value=None, visible=False),
                    )

            date_col_select.change(
                on_date_change_cb, 
                inputs=[date_col_select], 
                outputs=[date_range_label, date_min_dropdown, date_max_dropdown]
            )

            # --- Add numeric filter ---
            def add_numeric_filter_cb(state, col, min_choice, max_choice):
                """Add numeric filter."""
                print(f"\nDEBUG: add_numeric_filter_cb called")
                print(f"  col={col}, min={min_choice}, max={max_choice}")

                state = state or {"numerical": {}, "categorical": {}, "date": {}}

                if not col:
                    return state, gr.update(value="Select a numeric column first."), gr.update(value=state)

                # Parse choices (they come as strings)
                chosen_min = None
                chosen_max = None

                if min_choice:
                    try:
                        chosen_min = float(min_choice)
                    except:
                        chosen_min = pd.to_numeric(min_choice, errors="coerce")
                        if pd.isna(chosen_min):
                            chosen_min = None

                if max_choice:
                    try:
                        chosen_max = float(max_choice)
                    except:
                        chosen_max = pd.to_numeric(max_choice, errors="coerce")
                        if pd.isna(chosen_max):
                            chosen_max = None

                if chosen_min is None and chosen_max is None:
                    return state, gr.update(value="No numeric bounds provided."), gr.update(value=state)

                # Ensure order
                if chosen_min is not None and chosen_max is not None and chosen_min > chosen_max:
                    chosen_min, chosen_max = chosen_max, chosen_min

                state["numerical"][col] = {"min": chosen_min, "max": chosen_max}

                print(f"DEBUG: Added filter: {state['numerical'][col]}")

                return state, gr.update(value=f"Added numeric filter: {col} ∈ [{chosen_min}, {chosen_max}]"), gr.update(value=state)

            add_num_btn.click(
                add_numeric_filter_cb,
                inputs=[filters_state, num_col_select, num_min_dropdown, num_max_dropdown],
                outputs=[filters_state, apply_status, filters_display]
            )

            # --- Add categorical filter ---
            def add_categorical_filter_cb(state, col, selected_vals):
                """Add categorical filter."""
                state = state or {"numerical": {}, "categorical": {}, "date": {}}
                if not col:
                    return state, gr.update(value="Select categorical column first."), gr.update(value=state)
                if not selected_vals:
                    return state, gr.update(value="No values selected."), gr.update(value=state)
                state["categorical"][col] = list(selected_vals)
                return state, gr.update(value=f"Added categorical filter on {col}: {len(selected_vals)} values"), gr.update(value=state)

            add_cat_btn.click(
                add_categorical_filter_cb, 
                inputs=[filters_state, cat_col_select, cat_multiselect], 
                outputs=[filters_state, apply_status, filters_display]
            )

            # --- Add date filter ---
            def add_date_filter_cb(state, col, min_choice, max_choice):
                """Add date filter."""
                state = state or {"numerical": {}, "categorical": {}, "date": {}}
                if not col:
                    return state, gr.update(value="Select a date column first."), gr.update(value=state)

                chosen_start = min_choice or None
                chosen_end = max_choice or None

                # Validate and normalize
                if chosen_start:
                    try:
                        chosen_start = pd.to_datetime(chosen_start).strftime("%Y-%m-%d")
                    except:
                        return state, gr.update(value="Invalid start date."), gr.update(value=state)

                if chosen_end:
                    try:
                        chosen_end = pd.to_datetime(chosen_end).strftime("%Y-%m-%d")
                    except:
                        return state, gr.update(value="Invalid end date."), gr.update(value=state)

                if not chosen_start and not chosen_end:
                    return state, gr.update(value="No date bounds provided."), gr.update(value=state)

                if chosen_start and chosen_end and pd.to_datetime(chosen_start) > pd.to_datetime(chosen_end):
                    chosen_start, chosen_end = chosen_end, chosen_start

                state["date"][col] = {"start": chosen_start, "end": chosen_end}
                return state, gr.update(value=f"Added date filter: {col} between {chosen_start} and {chosen_end}"), gr.update(value=state)

            add_date_btn.click(
                add_date_filter_cb, 
                inputs=[filters_state, date_col_select, date_min_dropdown, date_max_dropdown], 
                outputs=[filters_state, apply_status, filters_display]
            )

            # --- Remove filter UI ---
            remove_kind = gr.Dropdown(label="Kind to remove", choices=["numerical", "categorical", "date"], value="numerical")
            remove_col = gr.Textbox(label="Column name to remove")
            remove_btn = gr.Button("Remove filter")

            def remove_filter_cb(state, kind, col):
                """Remove filter."""
                state = state or {"numerical": {}, "categorical": {}, "date": {}}
                if not col:
                    return state, gr.update(value="Provide column name to remove."), gr.update(value=state)
                state.get(kind, {}).pop(col, None)
                return state, gr.update(value=f"Removed filter {kind}:{col} if present."), gr.update(value=state)

            remove_btn.click(
                remove_filter_cb, 
                inputs=[filters_state, remove_kind, remove_col], 
                outputs=[filters_state, apply_status, filters_display]
            )

            # --- Apply filters ---
            def apply_filters_cb(state, selected_display_cols):
                """Apply filters."""
                df = dp.get_dataframe()
                if df is None:
                    return "No data loaded.", pd.DataFrame(), 0

                state = state or {"numerical": {}, "categorical": {}, "date": {}}
                num_filters = state.get("numerical", {})
                cat_filters = state.get("categorical", {})
                date_filters = state.get("date", {})

                try:
                    filtered_df, row_count = apply_all_filters(
                        df, 
                        num_filters=num_filters, 
                        cat_filters=cat_filters, 
                        date_filters=date_filters, 
                        display_columns=selected_display_cols or None
                    )

                    dp.set_filtered(filtered_df)
                    return f"Filtered rows: {row_count:,}", filtered_df.head(50), row_count

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return f"Error applying filters: {str(e)}", pd.DataFrame(), 0

            apply_filters_btn.click(
                apply_filters_cb, 
                inputs=[filters_state, display_cols], 
                outputs=[apply_status, filter_preview, row_count_box]
            )

            # Show live state
            filters_state.change(lambda s: s, inputs=[filters_state], outputs=[filters_display])

            # Wire up metadata and display-cols
            gen_meta_btn.click(
                inspect_dataset_for_ui_cb, 
                inputs=[], 
                outputs=[meta_output, num_col_select, cat_col_select, date_col_select]
            )
            gen_meta_btn.click(populate_display_cols_cb, inputs=[], outputs=[display_cols])

        # --- Visualization Tab ---
        with gr.Tab("Visualizations"):
            gr.Markdown("### Create Interactive Visualizations")
            gr.Markdown("Select visualization type and configure parameters below.")

            # ===== Visualization Type Selection =====
            viz_type = gr.Dropdown(
                choices=["Time Series", "Distribution", "Category", "Scatter/Heatmap"],
                label="📊 Visualization Type",
                value="Distribution",
                info="Choose the type of chart to create"
            )

            # ===== Common Controls (shown/hidden based on viz type) =====
            with gr.Row():
                with gr.Column(scale=1):
                    col_x = gr.Dropdown(
                        choices=[],
                        label="X-Axis Column",
                        info="Select column for X-axis",
                        interactive=True
                    )

                    col_y = gr.Dropdown(
                        choices=[],
                        label="Y-Axis Column",
                        info="Select column for Y-axis",
                        interactive=True,
                        visible=False
                    )

                    group_col = gr.Dropdown(
                        choices=[],
                        label="Group By Column (Optional)",
                        info="Split time series by category",
                        interactive=True,
                        visible=False
                    )

                with gr.Column(scale=1):
                    agg_method = gr.Dropdown(
                        choices=["count", "sum", "mean", "median"],
                        label="Aggregation Method",
                        value="sum",
                        info="How to aggregate values",
                        visible=False
                    )

                    chart_subtype = gr.Dropdown(
                        choices=["histogram", "box"],
                        label="Chart Type",
                        value="histogram",
                        info="Specific chart variant",
                        visible=True
                    )

                    freq = gr.Dropdown(
                        choices=[
                            ("Daily", "D"),
                            ("Weekly", "W"),
                            ("Monthly", "M"),
                            ("Quarterly", "Q"),
                            ("Yearly", "Y")
                        ],
                        label="Time Frequency",
                        value="D",
                        info="Resample frequency for time series",
                        visible=False
                    )

                    top_n = gr.Slider(
                        minimum=5,
                        maximum=50,
                        step=5,
                        value=20,
                        label="Top N Categories",
                        info="Show top N categories",
                        visible=False
                    )

                    corr_method = gr.Dropdown(
                        choices=["pearson", "spearman", "kendall"],
                        label="Correlation Method",
                        value="pearson",
                        visible=False
                    )

            # ===== Generate & Export Buttons =====
            with gr.Row():
                viz_btn = gr.Button("🎨 Generate Visualization", variant="primary", scale=2)
                refresh_cols_btn = gr.Button("🔄 Refresh Columns", scale=1)
                viz_export_btn = gr.Button("💾 Export as PNG", scale=1)

            # ===== Outputs =====
            viz_status = gr.Textbox(label="Status", interactive=False, visible=True)
            viz_output = gr.Plot(label="Visualization")
            viz_export_file = gr.File(label="Download PNG", visible=False)

            # Hidden state to store the actual figure object
            current_fig = gr.State(None)

            # ===== Helper Functions =====

            def get_dataframe_safe():
                """Safely get DataFrame with proper None checking."""
                filtered_df = dp.get_filtered()
                main_df = dp.get_dataframe()

                if filtered_df is not None:
                    return filtered_df
                elif main_df is not None:
                    return main_df
                else:
                    return None

            def get_column_lists():
                """Get lists of columns by type from current dataframe."""
                df = get_dataframe_safe()

                if df is None or df.empty:
                    return [], [], [], []

                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
                datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

                # Try to detect date-like string columns
                for col in df.select_dtypes(include=["object"]).columns:
                    if col in datetime_cols:
                        continue
                    try:
                        sample = df[col].dropna().head(100)
                        if len(sample) > 0:
                            parsed = pd.to_datetime(sample, errors="coerce")
                            if parsed.notna().sum() / len(sample) > 0.5:
                                datetime_cols.append(col)
                    except:
                        pass
                    
                all_cols = df.columns.tolist()
                return numeric_cols, categorical_cols, datetime_cols, all_cols

            def update_controls_for_viz_type(viz_type_val):
                """Update all controls based on selected visualization type."""
                num_cols, cat_cols, date_cols, all_cols = get_column_lists()

                if not all_cols:
                    empty_update = gr.update(choices=[], value=None, visible=False)
                    return [
                        empty_update,  # col_x
                        empty_update,  # col_y
                        empty_update,  # group_col
                        gr.update(visible=False),  # agg_method
                        empty_update,  # chart_subtype
                        gr.update(visible=False),  # freq
                        gr.update(visible=False),  # top_n
                        gr.update(visible=False),  # corr_method
                        "⚠️ No data loaded. Please upload data first.",  # status
                    ]

                if viz_type_val == "Time Series":
                    return [
                        gr.update(choices=date_cols, value=date_cols[0] if date_cols else None, label="📅 Date Column (X-axis)", visible=True),
                        gr.update(choices=num_cols, value=num_cols[0] if num_cols else None, label="📊 Value Column (Y-axis)", visible=True),
                        gr.update(choices=["None"] + cat_cols, value="None", label="🔀 Group By (Optional)", visible=True),
                        gr.update(visible=True, value="sum"),
                        gr.update(visible=False),
                        gr.update(visible=True, value="D"),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        "ℹ️ Select date and value columns, then click Generate",
                    ]

                elif viz_type_val == "Distribution":
                    return [
                        gr.update(choices=num_cols, value=num_cols[0] if num_cols else None, label="📊 Numeric Column", visible=True),
                        gr.update(visible=False, value=None),
                        gr.update(visible=False, value=None),
                        gr.update(visible=False),
                        gr.update(choices=["histogram", "box"], value="histogram", label="Distribution Type", visible=True),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        "ℹ️ Select a numeric column and click Generate",
                    ]

                elif viz_type_val == "Category":
                    return [
                        gr.update(choices=cat_cols, value=cat_cols[0] if cat_cols else None, label="📑 Category Column", visible=True),
                        gr.update(choices=["None"] + num_cols, value="None", label="📊 Value Column (optional)", visible=True),
                        gr.update(visible=False, value=None),
                        gr.update(visible=True, value="count"),
                        gr.update(choices=["bar", "pie"], value="bar", label="Chart Type", visible=True),
                        gr.update(visible=False),
                        gr.update(visible=True, value=20),
                        gr.update(visible=False),
                        "ℹ️ Select category column and click Generate",
                    ]

                elif viz_type_val == "Scatter/Heatmap":
                    return [
                        gr.update(choices=num_cols, value=num_cols[0] if num_cols else None, label="📊 X Column", visible=True),
                        gr.update(choices=num_cols, value=num_cols[1] if len(num_cols) > 1 else None, label="📊 Y Column", visible=True),
                        gr.update(visible=False, value=None),
                        gr.update(visible=False),
                        gr.update(choices=["scatter", "heatmap"], value="scatter", label="Chart Type", visible=True),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=True, value="pearson"),
                        "ℹ️ Configure scatter/heatmap and click Generate",
                    ]

                return [gr.update()] * 9

            def generate_visualization(viz_type_val, x_col, y_col, grp_col, agg, subtype, frequency, top, corr):
                """Generate the selected visualization."""
                df = get_dataframe_safe()

                if df is None or df.empty:
                    return None, None, "❌ No data available. Please load data first.", gr.update(visible=False)

                try:
                    if viz_type_val == "Time Series":
                        if not x_col or not y_col:
                            return None, None, "❌ Please select both date and value columns.", gr.update(visible=False)

                        group_column = None if (not grp_col or grp_col == "None") else grp_col

                        fig = make_time_series(
                            df,
                            date_col=x_col,
                            value_col=y_col,
                            agg=agg,
                            freq=frequency,
                            group_col=group_column
                        )

                        if group_column:
                            status = f"✅ Time series created: {y_col} over {x_col} grouped by {group_column}"
                        else:
                            status = f"✅ Time series created: {y_col} over {x_col}"

                    elif viz_type_val == "Distribution":
                        if not x_col:
                            return None, None, "❌ Please select a numeric column.", gr.update(visible=False)

                        fig = make_distribution(df, col=x_col, chart_type=subtype)
                        status = f"✅ Distribution created for {x_col}"

                    elif viz_type_val == "Category":
                        if not x_col:
                            return None, None, "❌ Please select a category column.", gr.update(visible=False)

                        value_column = None if (not y_col or y_col == "None") else y_col

                        fig = make_category_chart(
                            df,
                            cat_col=x_col,
                            agg_col=value_column,
                            agg=agg,
                            chart=subtype,
                            top_n=int(top)
                        )
                        status = f"✅ Category chart created for {x_col}"

                    elif viz_type_val == "Scatter/Heatmap":
                        if subtype == "scatter":
                            if not x_col or not y_col:
                                return None, None, "❌ Please select both X and Y columns for scatter plot.", gr.update(visible=False)
                            fig = make_scatter_or_heatmap(df, col_x=x_col, col_y=y_col, chart_type="scatter")
                            status = f"✅ Scatter plot created: {x_col} vs {y_col}"
                        else:
                            fig = make_scatter_or_heatmap(df, chart_type="heatmap", corr_method=corr)
                            status = f"✅ Correlation heatmap created ({corr} method)"

                    else:
                        return None, None, "❌ Unknown visualization type.", gr.update(visible=False)

                    # Return: figure for display, figure for state, status, file visibility
                    return fig, fig, status, gr.update(visible=True)

                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    print("=" * 60)
                    print("VISUALIZATION ERROR DETAILS:")
                    print("=" * 60)
                    print(error_details)
                    print("=" * 60)
                    return None, None, f"❌ Error: {str(e)}", gr.update(visible=False)

            def export_viz_to_png(stored_fig):
                """Export the current visualization to PNG."""
                if stored_fig is None:
                    return None, "❌ No visualization to export. Generate a chart first."

                try:
                    import tempfile
                    import os

                    # stored_fig is the actual plotly figure object
                    img_bytes = fig_to_png_bytes(stored_fig)

                    # Create a temporary file and write the bytes to it
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False, 
                        suffix='.png', 
                        prefix='visualization_'
                    )
                    temp_file.write(img_bytes)
                    temp_file.close()

                    # Return the file path (Gradio will handle the download)
                    return temp_file.name, "✅ Visualization exported successfully!"
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    print("=" * 60)
                    print("PNG EXPORT ERROR DETAILS:")
                    print("=" * 60)
                    print(error_details)
                    print("=" * 60)
                    return None, f"❌ Export failed: {str(e)}"

            # ===== Wire Up Events =====

            # Update controls when viz type changes
            viz_type.change(
                update_controls_for_viz_type,
                inputs=[viz_type],
                outputs=[col_x, col_y, group_col, agg_method, chart_subtype, freq, top_n, corr_method, viz_status]
            )

            # Refresh columns manually
            refresh_cols_btn.click(
                update_controls_for_viz_type,
                inputs=[viz_type],
                outputs=[col_x, col_y, group_col, agg_method, chart_subtype, freq, top_n, corr_method, viz_status]
            )

            # Generate visualization - now outputs to both viz_output AND current_fig state
            viz_btn.click(
                generate_visualization,
                inputs=[viz_type, col_x, col_y, group_col, agg_method, chart_subtype, freq, top_n, corr_method],
                outputs=[viz_output, current_fig, viz_status, viz_export_file]
            )

            # Export visualization - uses the stored figure from state
            viz_export_btn.click(
                export_viz_to_png,
                inputs=[current_fig],  # Use the state, not the Plot component
                outputs=[viz_export_file, viz_status]
            )

            def init_viz_dropdowns_on_load():
                """Initialize visualization dropdowns when data is first loaded."""
                return update_controls_for_viz_type(viz_type.value or "Distribution")
            
            # Wire it to trigger when upload happens
            upload_btn.click(
                init_viz_dropdowns_on_load,
                inputs=[],
                outputs=[col_x, col_y, group_col, agg_method, chart_subtype, freq, top_n, corr_method, viz_status]
            )
            
            sample1_btn.click(
                init_viz_dropdowns_on_load,
                inputs=[],
                outputs=[col_x, col_y, group_col, agg_method, chart_subtype, freq, top_n, corr_method, viz_status]
            )
            
            sample2_btn.click(
                init_viz_dropdowns_on_load,
                inputs=[],
                outputs=[col_x, col_y, group_col, agg_method, chart_subtype, freq, top_n, corr_method, viz_status]
            )

        # --- Insights ---
        with gr.Tab("Insights"):
            gr.Markdown("### 🔍 Automated Business Insights")
            gr.Markdown("AI-powered analysis that automatically identifies patterns, trends, and anomalies in your data.")

            with gr.Row():
                with gr.Column(scale=1):
                    # Configuration
                    gr.Markdown("#### ⚙️ Configuration")
                    date_col_select = gr.Dropdown(
                        choices=[],
                        label="Date Column (Optional)",
                        info="Select to enable trend analysis over time",
                        interactive=True
                    )

                    top_n_insight = gr.Slider(
                        minimum=3,
                        maximum=20,
                        step=1,
                        value=5,
                        label="Number of Records to Show",
                        info="How many top/bottom records to display"
                    )

                    with gr.Row():
                        generate_insights_btn = gr.Button("🔍 Generate Insights", variant="primary", scale=2)
                        refresh_insight_cols_btn = gr.Button("🔄 Refresh", scale=1)

                with gr.Column(scale=2):
                    # Quick summary with metrics
                    gr.Markdown("#### 📊 Dataset Overview")
                    with gr.Row():
                        total_rows_box = gr.Textbox(label="Total Rows", interactive=False)
                        total_cols_box = gr.Textbox(label="Total Columns", interactive=False)
                        missing_pct_box = gr.Textbox(label="Missing Data %", interactive=False)

            # Status
            insights_status = gr.Textbox(label="Status", interactive=False)

            # Tabbed insights display
            with gr.Tabs():
                # Tab 1: Executive Summary
                with gr.Tab("📋 Executive Summary"):
                    gr.Markdown("""
                    ### Key Findings
                    This tab provides a high-level overview of the most important insights from your data.
                    """)

                    exec_summary_md = gr.Markdown(value="*Generate insights to see summary*")

                # Tab 2: Top/Bottom Performers
                with gr.Tab("🏆 Top/Bottom Performers"):
                    gr.Markdown("""
                    ### Performance Analysis
                    Identifies the highest and lowest performing records in your dataset.
                    **Use Case:** Find best/worst selling products, highest/lowest revenue orders, etc.
                    """)

                    performer_column = gr.Dropdown(
                        choices=[],
                        label="📊 Select Column to Analyze",
                        info="Choose any numeric column"
                    )

                    performer_summary_md = gr.Markdown(value="")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 🔝 Top Performers")
                            top_performers_df = gr.Dataframe(label="Highest Values", interactive=False)
                        with gr.Column():
                            gr.Markdown("#### 🔻 Bottom Performers")
                            bottom_performers_df = gr.Dataframe(label="Lowest Values", interactive=False)

                # Tab 3: Trends Over Time
                with gr.Tab("📈 Trends"):
                    gr.Markdown("""
                    ### Trend Analysis
                    Analyzes how values change over time to identify growth or decline patterns.
                    **Use Case:** Track sales trends, monitor customer growth, identify seasonal patterns.
                    """)

                    trend_summary_md = gr.Markdown(value="*Select a date column in Configuration to enable trend analysis*")

                # Tab 4: Anomalies & Outliers
                with gr.Tab("⚠️ Anomalies"):
                    gr.Markdown("""
                    ### Outlier Detection
                    Identifies unusual values that deviate significantly from the norm using statistical methods (IQR).
                    **Use Case:** Detect fraudulent transactions, data entry errors, or exceptional cases.
                    """)

                    anomaly_column = gr.Dropdown(
                        choices=[],
                        label="📊 Select Column to Check",
                        info="Choose a numeric column to detect outliers"
                    )

                    anomaly_summary_md = gr.Markdown(value="")
                    outliers_df = gr.Dataframe(label="Outlier Records", interactive=False)

                # Tab 5: Correlations
                with gr.Tab("🔗 Correlations"):
                    gr.Markdown("""
                    ### Relationship Analysis
                    Discovers strong relationships between different numeric variables in your data.
                    **Use Case:** Understand what factors influence sales, find related metrics.

                    **Interpretation:**
                    - **Positive correlation (0.7 to 1.0):** Variables move together (e.g., quantity ↑ → revenue ↑)
                    - **Negative correlation (-0.7 to -1.0):** Variables move oppositely (e.g., discount ↑ → profit ↓)
                    """)

                    correlation_summary_md = gr.Markdown(value="")
                    correlation_df = gr.Dataframe(
                        label="Strong Correlations (|r| ≥ 0.7)",
                        interactive=False
                    )

                # Tab 6: Distribution Analysis
                with gr.Tab("📊 Distributions"):
                    gr.Markdown("""
                    ### Distribution Shape Analysis
                    Examines how values are spread across your data.
                    **Use Case:** Understand typical values, identify data skewness, spot unusual distributions.

                    **Interpretation:**
                    - **Symmetric:** Data is evenly distributed (normal pattern)
                    - **Right-skewed:** Most values are low, with some very high values (common in sales data)
                    - **Left-skewed:** Most values are high, with some very low values
                    """)

                    distribution_column = gr.Dropdown(
                        choices=[],
                        label="📊 Select Column",
                        info="Choose a numeric column"
                    )

                    distribution_summary_md = gr.Markdown(value="")

                # Tab 7: Category Insights
                with gr.Tab("📑 Categories"):
                    gr.Markdown("""
                    ### Categorical Analysis
                    Analyzes the distribution of categorical data (text fields like product types, regions, etc.).
                    **Use Case:** Find most popular categories, identify data imbalances.
                    """)

                    category_column = gr.Dropdown(
                        choices=[],
                        label="📑 Select Categorical Column"
                    )

                    category_summary_md = gr.Markdown(value="")
                    category_chart_df = gr.Dataframe(
                        label="Category Distribution",
                        interactive=False
                    )

            # Hidden state to store all insights
            insights_data = gr.State({})

            # ===== Helper Functions =====

            def get_dataframe_safe():
                """Safely get DataFrame."""
                filtered_df = dp.get_filtered()
                main_df = dp.get_dataframe()

                if filtered_df is not None:
                    return filtered_df
                elif main_df is not None:
                    return main_df
                else:
                    return None

            def refresh_insight_columns():
                """Refresh column dropdowns for insights."""
                df = get_dataframe_safe()

                if df is None or df.empty:
                    return [
                        gr.update(choices=[], value=None),  # date_col_select
                        gr.update(choices=[], value=None),  # performer_column
                        gr.update(choices=[], value=None),  # anomaly_column
                        gr.update(choices=[], value=None),  # distribution_column
                        gr.update(choices=[], value=None)   # category_column
                    ]

                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

                # Try to detect date columns
                date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
                for col in df.select_dtypes(include=["object"]).columns:
                    try:
                        sample = df[col].dropna().head(100)
                        if len(sample) > 0:
                            parsed = pd.to_datetime(sample, errors="coerce")
                            if parsed.notna().sum() / len(sample) > 0.5:
                                if col not in date_cols:  # Avoid duplicates
                                    date_cols.append(col)
                    except:
                        pass
                    
                return [
                    gr.update(choices=["None"] + date_cols, value="None", interactive=True),  # date_col_select - FIX: Added interactive=True
                    gr.update(choices=numeric_cols, value=numeric_cols[0] if numeric_cols else None, interactive=True),  # performer_column
                    gr.update(choices=numeric_cols, value=numeric_cols[0] if numeric_cols else None, interactive=True),  # anomaly_column
                    gr.update(choices=numeric_cols, value=numeric_cols[0] if numeric_cols else None, interactive=True),  # distribution_column
                    gr.update(choices=cat_cols, value=cat_cols[0] if cat_cols else None, interactive=True)   # category_column
                ]


            # Add auto-initialization for insights when data loads:
            def init_insight_dropdowns_on_load():
                """Initialize insight dropdowns when data is first loaded."""
                return refresh_insight_columns()


            def format_executive_summary(insights):
                """Create executive summary in markdown."""
                if not insights:
                    return "*No insights available*"

                summary = insights.get("summary", {})

                md = "## 📊 Data Overview\n\n"
                md += f"- **Dataset Size:** {summary.get('total_rows', 0):,} rows × {summary.get('total_columns', 0)} columns\n"
                md += f"- **Numeric Columns:** {summary.get('numeric_columns', 0)}\n"
                md += f"- **Missing Data:** {summary.get('missing_values', 0):,} values ({summary.get('missing_percent', 0)}%)\n\n"

                # Key findings
                md += "## 🔍 Key Findings\n\n"

                # Trends
                trends = insights.get("trends", {})
                if trends:
                    md += "### 📈 Trends\n"
                    for col, trend_data in list(trends.items())[:3]:
                        emoji = "📈" if trend_data.get("trend") == "increasing" else "📉" if trend_data.get("trend") == "decreasing" else "➡️"
                        md += f"- {emoji} **{col}:** {trend_data.get('message', '')}\n"
                    md += "\n"

                # Anomalies
                anomalies = insights.get("anomalies", {})
                if anomalies:
                    md += "### ⚠️ Anomalies Detected\n"
                    for col, anomaly_data in list(anomalies.items())[:3]:
                        count = anomaly_data.get("outlier_count", 0)
                        pct = anomaly_data.get("outlier_percent", 0)
                        md += f"- **{col}:** {count:,} outliers detected ({pct}% of data)\n"
                    md += "\n"

                # Correlations
                correlations = insights.get("correlations", [])
                if correlations:
                    md += "### 🔗 Strong Relationships Found\n"
                    for corr in correlations[:3]:
                        emoji = "🔵" if corr["correlation"] > 0 else "🔴"
                        md += f"- {emoji} **{corr['column1']}** ↔ **{corr['column2']}**: {corr['strength']} ({corr['correlation']:.2f})\n"
                    md += "\n"

                # Top performers
                top_bottom = insights.get("top_bottom", {})
                if top_bottom:
                    first_col = list(top_bottom.keys())[0]
                    data = top_bottom[first_col]
                    md += "### 🏆 Performance Highlights\n"
                    md += f"**{first_col}** - Top value: {data['top_values'][0] if data['top_values'] else 'N/A'}, "
                    md += f"Bottom value: {data['bottom_values'][0] if data['bottom_values'] else 'N/A'}\n\n"

                return md

            def generate_all_insights(date_col_val, top_n_val):
                """Generate comprehensive insights."""
                df = get_dataframe_safe()

                if df is None or df.empty:
                    return (
                        {},  # insights_data
                        "❌ No data available. Please load data first.",  # status
                        "", "", "",  # metrics boxes
                        "",  # exec_summary
                        gr.update(choices=[]),  # performer_column
                        gr.update(choices=[]),  # anomaly_column
                        gr.update(choices=[]),  # distribution_column
                        gr.update(choices=[])   # category_column
                    )

                try:
                    # Get numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

                    # Use date column if provided and not "None"
                    date_col_to_use = None if (not date_col_val or date_col_val == "None") else date_col_val

                    # Generate insights
                    insights = generate_insights(
                        df,
                        numeric_cols=numeric_cols,
                        date_col=date_col_to_use,
                        top_n=int(top_n_val)
                    )

                    # Format metrics
                    total_rows = f"{insights['summary']['total_rows']:,}"
                    total_cols = str(insights['summary']['total_columns'])
                    missing_pct = f"{insights['summary']['missing_percent']}%"

                    # Generate executive summary
                    exec_summary = format_executive_summary(insights)

                    status_msg = f"✅ Analysis complete! Found {len(insights.get('anomalies', {}))} columns with anomalies, "
                    status_msg += f"{len(insights.get('correlations', []))} strong correlations."

                    return (
                        insights,  # insights_data (state)
                        status_msg,  # status
                        total_rows, total_cols, missing_pct,  # metrics
                        exec_summary,  # exec_summary_md
                        gr.update(choices=numeric_cols, value=numeric_cols[0] if numeric_cols else None),  # performer_column
                        gr.update(choices=numeric_cols, value=numeric_cols[0] if numeric_cols else None),  # anomaly_column
                        gr.update(choices=numeric_cols, value=numeric_cols[0] if numeric_cols else None),  # distribution_column
                        gr.update(choices=list(insights.get("categories", {}).keys()), 
                                 value=list(insights.get("categories", {}).keys())[0] if insights.get("categories") else None)  # category_column
                    )

                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    print("=" * 60)
                    print("INSIGHTS ERROR:")
                    print("=" * 60)
                    print(error_details)
                    print("=" * 60)

                    return (
                        {}, f"❌ Error: {str(e)}", "", "", "", "",
                        gr.update(), gr.update(), gr.update(), gr.update()
                    )

            def show_top_bottom(insights, column):
                """Show top/bottom performers with interpretation."""
                if not insights or not column or column not in insights.get("top_bottom", {}):
                    return "", pd.DataFrame(), pd.DataFrame()

                data = insights["top_bottom"][column]

                # Create summary
                summary = f"### 📊 Analysis of **{column}**\n\n"
                summary += f"**Statistics:**\n"
                summary += f"- Average (Mean): **{data['mean']:,.2f}**\n"
                summary += f"- Median: **{data['median']:,.2f}**\n"
                summary += f"- Standard Deviation: **{data['std']:,.2f}**\n\n"

                if data['top_values']:
                    summary += f"**Interpretation:**\n"
                    summary += f"- Highest value: **{data['top_values'][0]:,.2f}**\n"
                    summary += f"- Lowest value: **{data['bottom_values'][0]:,.2f}**\n"
                    summary += f"- Range: **{data['top_values'][0] - data['bottom_values'][0]:,.2f}**\n"

                return summary, data["top"], data["bottom"]

            def show_trends(insights):
                """Format trend information."""
                if not insights or not insights.get("trends"):
                    return "*No trend analysis available. Select a date column in Configuration to enable.*"

                md = "## 📈 Trend Analysis Results\n\n"

                for col, trend_data in insights["trends"].items():
                    trend = trend_data.get("trend", "unknown")

                    if trend == "increasing":
                        emoji = "📈"
                        color = "green"
                    elif trend == "decreasing":
                        emoji = "📉"
                        color = "red"
                    else:
                        emoji = "➡️"
                        color = "gray"

                    md += f"### {emoji} **{col}**\n\n"
                    md += f"{trend_data.get('message', '')}\n\n"
                    md += f"**Details:**\n"
                    md += f"- Period: {trend_data.get('start_date', 'N/A')} to {trend_data.get('end_date', 'N/A')}\n"
                    md += f"- Start Value: **{trend_data.get('start_value', 0):,.2f}**\n"
                    md += f"- End Value: **{trend_data.get('end_value', 0):,.2f}**\n"
                    md += f"- Overall Change: **{trend_data.get('change_percent', 0):+.1f}%**\n\n"
                    md += "---\n\n"

                return md

            def show_anomalies(insights, column):
                """Show outliers with interpretation."""
                if not insights or not column or column not in insights.get("anomalies", {}):
                    return "*Select a column to view anomalies*", pd.DataFrame()

                data = insights["anomalies"][column]

                summary = f"### ⚠️ Anomaly Detection for **{column}**\n\n"
                summary += f"**Results:**\n"
                summary += f"- **{data['outlier_count']:,}** outliers detected ({data['outlier_percent']}% of data)\n"
                summary += f"- Normal range: **{data['lower_bound']:,.2f}** to **{data['upper_bound']:,.2f}**\n"
                summary += f"- Any values outside this range are considered unusual\n\n"

                if data['outlier_count'] > 0:
                    summary += f"**What does this mean?**\n"
                    summary += f"These records have unusually high or low values compared to the rest of your data. "
                    summary += f"They could represent errors, exceptional cases, or important outliers worth investigating.\n\n"

                    return summary, data["outliers"].head(20)
                else:
                    summary += "✅ No significant outliers detected. All values fall within the normal range.\n"
                    return summary, pd.DataFrame()

            def show_correlations(insights):
                """Format correlation insights."""
                if not insights or not insights.get("correlations"):
                    return "*No strong correlations found (threshold: |r| ≥ 0.7)*", pd.DataFrame()

                correlations = insights["correlations"]

                summary = f"### 🔗 Found {len(correlations)} Strong Correlations\n\n"
                summary += "**What this means:** These pairs of variables move together. When one changes, the other tends to change as well.\n\n"

                # Create DataFrame
                corr_data = []
                for corr in correlations:
                    corr_data.append([
                        corr["column1"],
                        corr["column2"],
                        f"{corr['correlation']:.3f}",
                        corr["strength"]
                    ])

                df = pd.DataFrame(corr_data, columns=["Column 1", "Column 2", "Correlation", "Relationship Type"])

                return summary, df

            def show_distribution(insights, column):
                """Show distribution analysis."""
                if not insights or not column or column not in insights.get("distributions", {}):
                    return "*Select a column to analyze distribution*"

                data = insights["distributions"][column]

                md = f"### 📊 Distribution Analysis: **{column}**\n\n"
                md += f"**Key Statistics:**\n"
                md += f"- Mean: **{data['mean']:,.2f}**\n"
                md += f"- Median: **{data['median']:,.2f}**\n"
                md += f"- Standard Deviation: **{data['std']:,.2f}**\n"
                md += f"- Range: **{data['min']:,.2f}** to **{data['max']:,.2f}**\n\n"

                md += f"**Shape:** {data['skewness_desc']}\n\n"

                md += "**Interpretation:**\n"
                if abs(data['skewness']) < 0.5:
                    md += "- Data is fairly balanced around the average\n"
                elif data['skewness'] > 0:
                    md += "- Most values are on the lower end with some very high values\n"
                    md += "- This is common for metrics like sales, income, or prices\n"
                else:
                    md += "- Most values are on the higher end with some very low values\n"

                return md

            def show_category_info(insights, column):
                """Show category analysis."""
                if not insights or not column or column not in insights.get("categories", {}):
                    return "*Select a categorical column*", pd.DataFrame()

                cat_data = insights["categories"][column]

                summary = f"### 📑 Category Analysis: **{column}**\n\n"
                summary += f"**Overview:**\n"
                summary += f"- Total unique values: **{cat_data['unique_count']}**\n"
                summary += f"- Most common: **{cat_data['most_common']}** "
                summary += f"({cat_data['most_common_count']:,} occurrences, {cat_data['most_common_percent']}%)\n\n"

                # Create DataFrame
                top_cats = []
                for cat, count in cat_data["top_categories"].items():
                    pct = cat_data["top_percentages"].get(cat, 0)
                    top_cats.append([cat, f"{count:,}", f"{pct}%"])

                df = pd.DataFrame(top_cats, columns=["Category", "Count", "Percentage"])

                return summary, df

            # ===== Wire Up Events =====

            # Refresh columns
            refresh_insight_cols_btn.click(
                refresh_insight_columns,
                inputs=[],
                outputs=[date_col_select, performer_column, anomaly_column, distribution_column, category_column]
            )

            # Generate insights
            generate_insights_btn.click(
                generate_all_insights,
                inputs=[date_col_select, top_n_insight],
                outputs=[
                    insights_data, insights_status,
                    total_rows_box, total_cols_box, missing_pct_box,
                    exec_summary_md,
                    performer_column, anomaly_column, distribution_column, category_column
                ]
            )

            # Show top/bottom performers
            performer_column.change(
                show_top_bottom,
                inputs=[insights_data, performer_column],
                outputs=[performer_summary_md, top_performers_df, bottom_performers_df]
            )

            # Show trends (triggered when insights are generated)
            insights_data.change(
                show_trends,
                inputs=[insights_data],
                outputs=[trend_summary_md]
            )

            # Show anomalies
            anomaly_column.change(
                show_anomalies,
                inputs=[insights_data, anomaly_column],
                outputs=[anomaly_summary_md, outliers_df]
            )

            # Show correlations (triggered when insights are generated)
            insights_data.change(
                show_correlations,
                inputs=[insights_data],
                outputs=[correlation_summary_md, correlation_df]
            )

            # Show distribution
            distribution_column.change(
                show_distribution,
                inputs=[insights_data, distribution_column],
                outputs=[distribution_summary_md]
            )

            # Show category info
            category_column.change(
                show_category_info,
                inputs=[insights_data, category_column],
                outputs=[category_summary_md, category_chart_df]
            )
        # --- Export Filtered Data ---
        with gr.Tab("Export Data"):
            gr.Markdown("### 💾 Export Your Data")
            gr.Markdown("Download the filtered dataset as a CSV file.")

            export_status = gr.Textbox(label="Export Status", interactive=False)
            export_btn = gr.Button("📥 Download Filtered Data as CSV", variant="primary")
            export_file = gr.File(label="Download File")

            gr.Markdown("""
            **What gets exported:**
            - All rows from your current filtered dataset
            - CSV format compatible with Excel, Google Sheets
            """)

            export_btn.click(export_filtered_data, inputs=[], outputs=[export_file, export_status])

    return demo


if __name__ == "__main__":
    demo = create_dashboard()
    demo.launch()
