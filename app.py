import subprocess
import sys

# Force install scikit-learn if not found
try:
    import sklearn
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    import sklearn  # Import again after installation

import gradio as gr
import pandas as pd
import pickle

# Load the pre-trained classification model (for Fire Occurrence)
with open('best_classification_model.pkl', 'rb') as clf_model_file:
    classification_model = pickle.load(clf_model_file)

# Load the pre-trained regression model (for Fire Size, Duration, and Suppression Cost)
with open('best_regression_model.pkl', 'rb') as reg_model_file:
    regression_model = pickle.load(reg_model_file)


def predict_wildfire(Temp, Humid, Wind, Rain, Fuel, Veget, Slope, Region):
    # Creating input DataFrame for the model
    input_data = pd.DataFrame({
        'Temperature (°C)': [Temp],
        'Humidity (%)': [Humid],
        'Wind Speed (km/h)': [Wind],
        'Rainfall (mm)': [Rain],
        'Fuel Moisture (%)': [Fuel],
        'Vegetation Type': [Veget],
        'Slope (%)': [Slope],
        'Region': [Region]
    })

    # One-hot encode the input data (ensure it matches the training data)
    input_encoded = pd.get_dummies(input_data)

     # Align columns with the training data for both models
    required_columns_clf = classification_model.feature_names_in_  # Get classification model features
    required_columns_reg = regression_model.feature_names_in_      # Get regression model features
     # Ensure columns align with both models
    for col in required_columns_clf:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded_clf = input_encoded[required_columns_clf]

    for col in required_columns_reg:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded_reg = input_encoded[required_columns_reg]

    # 1. Predict Fire Occurrence using the classification model
    fire_occurrence_prediction = classification_model.predict(input_encoded_clf)[0]
    fire_occurrence = 'Yes' if fire_occurrence_prediction == 1 else 'No'

    # 2. Predict Fire Size, Duration, and Suppression Cost using the regression model
    reg_predictions = regression_model.predict(input_encoded_reg)[0]
    fire_size = reg_predictions[0]
    fire_duration = reg_predictions[1]
    suppression_cost = reg_predictions[2]

    return fire_occurrence, fire_size, fire_duration, suppression_cost

# Gradio Interface using components
interface = gr.Interface(
    fn=predict_wildfire,
    inputs=[
        gr.Slider(minimum=0, maximum=100, step=0.1, label="Temperature (°C)"),
        gr.Slider(minimum=0, maximum=100, step=0.1, label="Humidity (%)"),
        gr.Slider(minimum=0, maximum=100, step=0.1, label="Wind Speed (km/h)"),
        gr.Slider(minimum=0, maximum=100, step=0.1, label="Rainfall (mm)"),
        gr.Slider(minimum=0, maximum=100, step=0.1, label="Fuel Moisture (%)"),
        gr.Dropdown(['Grassland', 'Forest', 'Shrubland'], label="Vegetation Type"),
        gr.Slider(minimum=0, maximum=100, step=0.1, label="Slope (%)"),
        gr.Dropdown(['South', 'East', 'North', 'West'], label="Region")
    ],
    outputs=[gr.Textbox(label="Predict Wildfire Occurrence"),
            gr.Number(label="Predicted Fire Size (hectares)"),
            gr.Number(label="Predicted Fire Duration (hours)"),
            gr.Number(label="Predicted Suppression Cost ($)")
    ],
    title="Predict Wildfire Occurrence and Outcomes"
)

if __name__ == "__main__":
    interface.launch()
