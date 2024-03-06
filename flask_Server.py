# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:33:55 2024

@author: KITCOOP
"""

from flask import Flask, render_template, Response, request
import pathlib
import cv2 #pip install opencv-python
import numpy as np
from tensorflow import keras

app= Flask(__name__)
name="test flask"

static= 'static/'

model= keras.models.load_model("model/numbermnist.h5")
def prediction(filename):
    img1= cv2.imread(filename) #1. read file 원본 이미지(28,28,3)
    img2= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #2. shape 수정(28,28)
    img3= np.expand_dims(img2, axis=0) #3 (1,28,28)
    print(str(img1.shape)+ "->" + str(img2.shape)+ "->" +str(img3.shape))
    print(str(img1.shape)+ "->" + str(img2.shape)+ "->" +str(img3.shape))
    
    img4= img3.reshape(1,28*28)/255
    predictions= model.predict(img4)
    print(predictions)
    maxno= np.argmax(predictions[0])
    print(maxno)
    return maxno


@app.route('/', methods=['GET', 'POST'])
def index():
    current_dir= pathlib.Path().absolute()
    print(current_dir)
    if request.method !=  'POST':
        return render_template('index.html', name=name)
    else :
        file= request.files['upload']
        filename= file.filename
        file.save(static+filename)
        maxno= prediction(static + filename)
        return render_template('index.html', image_file=filename, num= maxno)

if __name__ == '__main__':
    app.run(debug= True, port="5002")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    