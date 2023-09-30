from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import pickle

app = Flask(__name__)

loaded_model = pickle.load(open("model.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosporus'])
    K = int(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    prediction_proba = loaded_model.predict_proba(single_pred)

    crop_dict = {1: ("Rice", "rice.jpg"), 2: ("Maize", "maize.jpg"), 3: ("Jute", "jute.jpg"), 4: ("Cotton", "cotton.jpg"),
                 5: ("Coconut", "coconut.jpg"), 6: ("Papaya", "papaya.jpg"), 7: ("Orange", "orange.jpg"),
                 8: ("Apple", "apple.jpg"), 9: ("Muskmelon", "muskmelon.jpg"), 10: ("Watermelon", "watermelon.jpg"),
                 11: ("Grapes", "grapes.jpg"), 12: ("Mango", "mango.jpg"), 13: ("Banana", "banana.jpg"),
                 14: ("Pomegranate", "pomegranate.jpg"), 15: ("Lentil", "lentil.jpg"), 16: ("Blackgram", "blackgram.jpg"),
                 17: ("Mungbean", "mungbean.jpg"), 18: ("Mothbeans", "mothbeans.jpg"), 19: ("Pigeonpeas", "pigeonpeas.jpg"),
                 20: ("Kidneybeans", "kidneybeans.jpg"), 21: ("Chickpea", "chickpea.jpg"), 22: ("Coffee", "coffee.jpg")}

    crop_predictions = []
    for class_idx, prob in enumerate(prediction_proba[0]):
        crop_predictions.append((class_idx + 1, prob))

    crop_predictions.sort(key=lambda x: x[1], reverse=True)

    top_n = 2

    result = "Top {} crops to be cultivated:<br>".format(top_n)
    for i in range(top_n):
        if crop_predictions[i][0] in crop_dict:
            crop, image = crop_dict[crop_predictions[i][0]]
            result += '<div style="display: inline-block; margin: 10px; text-align: center;">'
            result += '<img src="/static/{}" width="200" height="200">'.format(image)
            result += "<br>{} with a probability of {:.2f}%".format(crop, crop_predictions[i][1] * 100)
            result += '</div>'

    return render_template('home.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
