# House Price Prediction Model

This repository contains a machine learning model for predicting house prices based on input data. The model is built using Python and utilizes the scikit-learn library for machine learning tasks.

## Dataset
The dataset used for training and testing the model is stored in the file `melb_data.csv`. This dataset contains various features related to houses such as number of rooms, bedrooms, bathrooms, land size, building area, year built, latitude, longitude, and the corresponding house prices.

## Usage
To use the model, follow these steps:

1. Clone the repository to your local machine.
2. Ensure you have Python and the necessary dependencies installed. You can install them using pip:
   ```
   pip install pandas scikit-learn
   ```
3. Run the provided Python script to train the model and make predictions.

## Model Training
The model is trained using a Random Forest Regressor, a popular ensemble learning method for regression tasks. The following steps are involved in training the model:

1. Load the dataset using pandas.
2. Clean the dataset by removing any rows with missing values.
3. Select relevant features (independent variables) and the target variable (house prices).
4. Split the dataset into training and validation sets.
5. Train the Random Forest Regressor model on the training data.

## Evaluation
After training the model, predictions are made on the validation set, and the Mean Absolute Error (MAE) is calculated to evaluate the model's performance. The MAE provides an indication of how accurate the predictions are in terms of the actual house prices.

## Example
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load the dataset
train_test_data = "melb_data.csv"
home_data = pd.read_csv(train_test_data)

# Clean the dataset
home_data = home_data.dropna(axis=0)

# Select features and target variable
fields = ['Rooms', 'Bedroom2', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = home_data[fields]
y = home_data.Price

# Split the dataset into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Train the Random Forest Regressor model
hp_model = RandomForestRegressor(random_state=1)
hp_model.fit(train_X, train_y)

# Make predictions
predictions = hp_model.predict(val_X)

# Evaluate the model
print("Mean Absolute Error:", mean_absolute_error(predictions, val_y))
```

## Note
Ensure that you have a Jupyter environment set up to run the provided code. If you encounter any issues or have questions, feel free to open an issue in this repository.

Happy predicting! 🏡💰
