This project implements a machine learning system to predict wildfire occurrence, fire size, fire duration, and suppression costs. It includes both classification and regression models, which were trained on a prepared dataset with key environmental and regional factors.
Project Files
  •	best_classification_model.pkl: The best-performing classification model, saved for predicting wildfire occurrence.
  •	best_regression_model.pkl: The best-performing regression model, saved for predicting fire size, duration, and suppression cost.
  
  •	label_encoder.pkl: (Not directly relevant in this script but often useful if categorical labels were encoded for the dataset).
  •	dataset.csv: Input dataset (replace with the actual dataset name) containing historical data on fire incidents and environmental factors.
Project Workflow
1. Data Preprocessing
  •	Features and Labels: The dataset includes:
    o	Classification target: Fire Occurrence (whether a wildfire will occur).
    o	Regression targets: Fire Size (hectares), Fire Duration (hours), and Suppression Cost ($).
    o	Input features: Environmental and geographical variables.
  •	Handling Missing Data: Rows with missing values in the classification target were removed.
  •	One-Hot Encoding: Categorical features were one-hot encoded to enable machine learning models to interpret the data.
  •	Train-Test Split: The dataset is divided into 80% training data and 20% test data to evaluate model performance.
  •	Standardization: The feature data was standardized for use with Support Vector Machine (SVM) models, as SVMs generally require scaled input.
2. Model Training and Evaluation
Classification Models
A variety of classification models were trained to predict Fire Occurrence:
  •	Random Forest Classifier
  •	Logistic Regression
  •	Decision Tree Classifier
  •	Support Vector Machine (SVM)
Each model was evaluated using Root Mean Squared Error (RMSE) to measure the accuracy of the predictions. The model with the lowest RMSE on the test data was selected as the best classification model.
Regression Models
For multi-output regression to predict fire attributes, a Random Forest Regressor was implemented. This model was trained to predict Fire Size, Fire Duration, and Suppression Cost simultaneously.
Each regression model was evaluated using RMSE, with the model showing the best performance selected for deployment.
3. Best Model Selection and Saving
  •	The best classification model was saved as best_classification_model.pkl.
  •	The best regression model was saved as best_regression_model.pkl.
Model Performance Summary
  •	Each model’s RMSE score on the test dataset was printed in the console to compare and select the best-performing models for classification and regression tasks.
How to Use
1.	Run the Script: Ensure the dataset and script are in the same directory. Running the script will automatically train the models, evaluate their performance, and save the best models.
2.	Model Deployment:
    o	Classification: The best_classification_model.pkl can be used to predict if a wildfire will occur given new environmental data.
    o	Regression: The best_regression_model.pkl can be used to estimate fire size, duration, and suppression costs based on input variables.
