

import flask
from flask import Flask,render_template,request
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from keras.models import load_model
from scipy import misc

import tensorflow as tf
global graph,classifier
graph = tf.get_default_graph()

classifier=load_model('my_model.h5')

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
	return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':
        file = request.files['image']
        if not file: 
            return render_template('index.html', label="No file")
        img = misc.imread(file)
        img = resize(img,(64,64), mode='constant')
        img = np.reshape(img,(1,64,64,3))
        with graph.as_default():
        	prediction = classifier.predict_classes(img)
        	label = int(prediction[0][0])
        	return render_template('index.html', label=label)


if __name__=='__main__':
	app.run(host='0.0.0.0',port=8000,debug=True)