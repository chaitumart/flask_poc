import flask
from flask import Flask, render_template, request
import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as np


'''
with ZipFile(file_name, 'r') as zip:
  zip.extractall()
  print('Done')
!mkdir static
cd /content/static
!mkdir images
cd ..
'''


app = Flask(__name__)
image_folder = os.path.join('static', 'images')
app.config["UPLOAD_FOLDER"] = image_folder


#model = load_model('my_model.h5')
input_shape = (224, 224, 3)
# ResNet50 is trained on color images with 224x224 pixels
import tensorflow.keras.applications.resnet50 as resnet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
model = resnet50.ResNet50(weights='imagenet',input_shape=input_shape)


def predict_label(img_path):

  # Load image
  img = image.load_img(img_path, target_size=input_shape)
  x = image.img_to_array(img)
  x= np.array([x])

  #preprocess it
  x = preprocess_input(x) # resnet related preprocessing
  #x = x.reshape(1, 100,100,3)

  # This is the inference time. Given an instance, it produces the predictions.
  preds = model.predict(x)
  #model.predict_classes(x)
  predictions = decode_predictions(preds, top=3)[0] # resnet prediction

  return predictions


 # routes
@app.route('/', methods=['GET'])
def home():
  return render_template('index.html')

@app.route('/', methods=['POST'])
def get_output():
  img = request.files['imagefile']
  img_path = '/content/static/images/' + img.filename
  img.save(img_path)
  p = predict_label(img_path)
  pic = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)

  #return render_template("index.html", prediction = p[0][1], img_path = img_path)
  return render_template('index.html', user_image=pic, prediction_text= 'Prediction : ' + p[0][1] )



 if __name__=='__main__':
  app.run()
