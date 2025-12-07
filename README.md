---
title: BI Dashboard
emoji: 📉
colorFrom: yellow
colorTo: pink
sdk: gradio
sdk_version: 6.0.2
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# BI_dashoard

Final Project: Interactive Business Intelligence Dashboard


Business Intelligence Dashboard (Gradio App)

An interactive Business Intelligence Dashboard built with Python, Gradio, and Pandas, designed to help non-technical users upload datasets, explore data visually, apply filters, and generate actionable insights through an easy-to-use web interface.

This application allows users to perform full data exploration without writing code, making it ideal for analytics teams, managers, and business stakeholders.

## Live Demo (Hugging Face Spaces)

🔗 Live App:
https://huggingface.co/spaces/<username>/<space-name>

## Features
1. Upload Any CSV File

Drag-and-drop file uploader

Automatic schema detection

Summary of column types (numeric, categorical, datetime)

2. Dynamic Filtering System

Filters are automatically generated based on column types:

Column Type	Filter Type
Numeric	Min/Max sliders
Categorical	Multi-select dropdown
Datetime	Date range picker

Filtering is fast, efficient, and works on all datasets.

3. Interactive Visualizations

Users can create a variety of charts:

Time Series

Distributions

Category-based bar charts

Scatterplots or heatmaps

Automatic chart selection based on column combinations

All charts are rendered with Matplotlib/Plotly and exportable as PNG.

4. Auto Insights Engine

The built-in insights module generates high-level findings automatically:

Trends

Correlations

Outliers

Category comparisons

This allows non-technical users to instantly understand their data.

5. Real-Time Data Preview

Table view of uploaded dataset

Table view of filtered dataset

Row count indicators

6. Clean, Modern Gradio UI

Well-organized layout

Tabs for Upload, Filter & Explore, and Insights

Works on Hugging Face Spaces with zero setup

## Project Structure
.
├── app.py                     # Main Gradio application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── utils/
│   ├── filtering.py           # Column detection & filtering logic
│   ├── visualizations.py      # Chart generation utilities
│   ├── insights.py            # Auto insights generator
│   └── data_processor.py      # Data cleaning and loading helpers
└── sample_data/               # Example CSV for testing

## Installation (Local Development)
1. Clone the repository
git clone https://github.com/ArzooMJ/BI_dashoard
cd <repo-name>

2. Install dependencies
pip install -r requirements.txt

3. Run the app
python app.py


The app will open at:

http://localhost:7860

## Deployment to Hugging Face Spaces

This app is fully compatible with Gradio Spaces.

Steps:

Go to: https://huggingface.co/spaces

Create a new Space → choose Gradio

Link your GitHub repository

Hugging Face auto-builds and hosts your app

Your live link becomes available instantly


## Technologies Used

Python 3.10+

Gradio

Pandas

NumPy

Matplotlib

Plotly


