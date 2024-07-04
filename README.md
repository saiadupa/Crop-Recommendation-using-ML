# Crop Recommendation System using Machine Learning

This project predicts the best crops to be cultivated based on various environmental factors such as Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall. The application uses a trained machine learning model to provide crop recommendations.

## Repository

You can find the source code and resources for this project on GitHub:
[CROP RECOMMENDATION SYSTEM REPOSITORY](https://github.com/saiadupa/Crop-Recommendation-using-ML.git)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/saiadupa/Crop-Recommendation-using-ML.git
    cd Crop-Recommendation-using-ML
    ```

2. Run the Flask application:

    ```bash
    python app.py
    ```

## Dataset

The dataset used for training the model is `Crop_recommendation.csv`, which contains the following features:
- `N` - Nitrogen
- `P` - Phosphorus
- `K` - Potassium
- `temperature` - Temperature
- `humidity` - Humidity
- `ph` - pH level
- `rainfall` - Rainfall
- `label` - Crop label

## Model Training

1. **Data Preprocessing**:
   - The dataset is read using pandas.
   - Features and labels are separated.
   - Data is split into training and testing sets.
   - Features are scaled using `StandardScaler`.

2. **Model Selection**:
   - Various models are evaluated: LDA, Logistic Regression, Naive Bayes, SVM, KNN, Decision Tree, Random Forest, Bagging, AdaBoost, Gradient Boosting, and Extra Trees.
   - Random Forest Classifier is chosen due to its high accuracy.

3. **Model Training**:
   - The chosen model is trained on the training set.
   - The trained model is saved using pickle for later use.

## Flask Application

1. **Home Route**:
   - Renders the home page with a form to input environmental parameters.

2. **Predict Route**:
   - Takes input from the form.
   - Uses the trained model to predict the probability of each crop.
   - Displays the top 2 crops along with their images and probabilities.

## Predictive Function

A predictive function `predict_crop` is defined to predict the best crop based on input values.

```python
def predict_crop(N, P, K, temperature, humidity, pH, rainfall):
    input_values = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
    prediction = rdf.predict(input_values)
    return prediction[0]
```

## Usage Example

```python
N = 21
P = 26
K = 27
tem = 27.003155
humidity = 47.675254
ph = 5.699587
rainfall = 95.851183

pred = predict_crop(N, P, K, tem, humidity, ph, rainfall)

if pred == 1:
    print("Rice is the best crop to be cultivated right there")
elif pred == 2:
    print("Maize is the best crop to be cultivated right there")
# ... other conditions
```

## Contributing

Feel free to fork this repository and contribute by submitting a pull request. For major changes, please open an issue first to discuss what you would like to change.
