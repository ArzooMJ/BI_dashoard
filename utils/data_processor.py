import pandas as pd
import json
import gradio as gr

class DataProcessor:
    def __init__(self):
        self.df = None                # Original dataframe
        self.filtered_df = None       # Filtered dataframe
        self.df_state = gr.State(None)

    # 1. LOAD & STORE DATA

    def load_file(self, file):
        """Load CSV or Excel and return DataFrame."""

        if file is None:
            raise ValueError("No file uploaded.")

        filename = file.name.lower()

        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(file.name)
            elif filename.endswith((".xls", ".xlsx")):
                df = pd.read_excel(file.name)
            else:
                raise ValueError("Unsupported file type. Upload CSV or Excel.")

            if df.empty:
                raise ValueError("Uploaded file is empty.")

            self.df = df
            self.filtered_df = df.copy()

            return df

        except Exception as e:
            raise ValueError(f"Failed to load file: {str(e)}")

    def set_dataframe(self, df):
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("Invalid dataframe provided")
        self.df = df
        self.filtered_df = df.copy()

    def get_dataframe(self):
        return self.df

    def get_filtered(self):
        return self.filtered_df
    
    def set_filtered(self, df_filtered):
        if df_filtered is None:
            raise ValueError("Filtered dataframe cannot be none")
        self.filtered_df=df_filtered

    def dataset_shape(self):
        if self.df is None:
            return None
        return list(self.df.shape)
    
    def columns_list(self):
        if self.df is None:
            return None
        return list(self.df.columns)
    
    def column_types(self):
        if self.df is None:
            return None
        return self.df.dtypes.astype(str).to_dict()
    
    def preview(self, n=10):
        if self.df is None:
            return None
        return self.df.head(n)

    # 2. SUMMARY STATISTICS

    def summary_statistics(self):
        """Return numeric and categorical summaries."""
        if self.df is None:
            return None, None

        # Numeric
        desc = self.df.describe(include="number").T

        # Categorical
        cat_cols = self.df.select_dtypes(include="object")
        if len(cat_cols.columns)>0:
            cat_summary = pd.DataFrame({
                "unique_values": cat_cols.nunique(),
                "most_frequent": cat_cols.mode().iloc[0]
            })
        else:
            cat_summary=pd.DataFrame()

        return desc, cat_summary

    def missing_report(self):
        """Return missing value counts."""
        if self.df is None:
            return None
        miss = self.df.isna().sum().to_frame(name="missing_count")
        miss["missing_percent"] = (miss["missing_count"] / len(self.df)) * 100
        return miss

    def correlation_matrix(self):
        """Correlation matrix only for numeric columns."""
        if self.df is None:
            return None
        return self.df.select_dtypes(include="number").corr()

    # ----------------------------------------------------
    # 3. FILTERING
    # ----------------------------------------------------
    def apply_filters(self, filters_json):
        """
        Filters JSON format example:
        {
            "age": {"min": 20, "max": 40},
            "gender": ["Male", "Female"],
            "date": {"start": "2020-01-01", "end": "2020-12-31"}
        }
        """
        if self.df is None:
            raise ValueError("No data loaded.")

        try:
            filters = json.loads(filters_json)
        except:
            raise ValueError("Invalid JSON. Please check your filter format.")

        df = self.df.copy()

        for col, rule in filters.items():

            # ---- Numeric Range ----
            if isinstance(rule, dict) and "min" in rule and "max" in rule:
                df = df[(df[col] >= rule["min"]) & (df[col] <= rule["max"])]

            # ---- Date Range ----
            elif isinstance(rule, dict) and "start" in rule and "end" in rule:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df = df[(df[col] >= pd.to_datetime(rule["start"])) &
                        (df[col] <= pd.to_datetime(rule["end"]))]

            # ---- Categorical List ----
            elif isinstance(rule, list):
                df = df[df[col].isin(rule)]

        self.filtered_df = df
        return df

    # ----------------------------------------------------
    # 4. COL TYPE HELPERS
    # ----------------------------------------------------
    def numeric_columns(self):
        if self.df is None:
            return []
        return self.df.select_dtypes(include="number").columns.tolist()

    def categorical_columns(self):
        if self.df is None:
            return []
        return self.df.select_dtypes(include="object").columns.tolist()
