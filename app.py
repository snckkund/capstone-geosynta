from flask import Flask, render_template, jsonify
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
import os

app = Flask(__name__)

# Load the predictions data
predictions_df = pd.read_csv('notebooks/output/future_predictions.csv', index_col=0)
predictions_df.index = pd.date_range(start='2025-01-31', end='2025-12-31', freq='M')

# Convert humidity to percentage
predictions_df['Humidity'] = predictions_df['Humidity'] * 100

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_predictions')
def get_predictions():
    # Convert predictions to JSON format
    predictions_json = {}
    for column in predictions_df.columns:
        predictions_json[column] = {
            'dates': predictions_df.index.strftime('%Y-%m-%d').tolist(),
            'values': predictions_df[column].tolist()
        }
    return jsonify(predictions_json)

@app.route('/get_metrics')
def get_metrics():
    # Calculate some basic metrics for each variable
    metrics = {}
    for column in predictions_df.columns:
        metrics[column] = {
            'mean': round(predictions_df[column].mean(), 2),
            'max': round(predictions_df[column].max(), 2),
            'min': round(predictions_df[column].min(), 2)
        }
    return jsonify(metrics)

if __name__ == '__main__':
    app.run(debug=True)
