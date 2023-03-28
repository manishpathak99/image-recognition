from flask import Flask, request
import os
import numpy as np
import tensorflow as tf
import requests
import json
from werkzeug.utils import secure_filename
from PIL import Image

from keras.models import load_model

app = Flask(__name__)
# Define the path to the pre-trained model file
model_path = 'model.h5'
model = load_model(model_path)

@app.route('/')
def index():
  return 'Server working !'

@app.route('/identify-image', methods=['POST'])
def identify_image():
    # Load the image and preprocess it
    file = request.files.get('image')
    img = Image.open(file)
    img = img.resize((224, 224))
    # img = tf.keras.preprocessing.image.load_img("/"+filename, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    
    #convert string data to numpy array
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Make a prediction using the trained model
    prediction = model.predict(img)
    print(prediction)

    # Get the predicted class label
    class_names = ['apple_pie', 'pizza', 'spaghetti_bolognese']
    predicted_class_idx = np.argmax(prediction, axis=-1)[0]
    predicted_class_label = class_names[predicted_class_idx]

    result = [{predicted_class_label:""}]
    
    return result

@app.route('/get-nutrition-data/<food_name>')
def get_nutrition_data(food_name):
    url = "https://edamam-recipe-search.p.rapidapi.com/search"
    querystring = {"q":str(food_name)}

    headers = {
	    "X-RapidAPI-Key": "b19894b9a8mshd86c43b6a6d4d05p115555jsnc157d4987e5c",
	    "X-RapidAPI-Host": "edamam-recipe-search.p.rapidapi.com"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    data = json.loads(response.text)['hits'][0]['recipe']

    totalCalories = data['calories']
    totalDaily = data['totalDaily']
    totalNutrients = data['totalNutrients']

    response = {'totalCalories':totalCalories, 'totalDaily':totalDaily, 'totalNutrients':totalNutrients}

    return response