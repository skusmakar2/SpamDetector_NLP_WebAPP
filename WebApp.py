#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 01:00:50 2020

@author: shitanshu
"""

from flask import Flask,render_template,request
import pickle

# load the model from disk
filename = 'SpamDetectorModel.pkl'
neigh = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('VectorizerFile.pkl','rb'))
DeployApp = Flask(__name__)

@DeployApp.route('/')
def home():
	return render_template('index.html')

@DeployApp.route('/predict',methods=['POST'])

def predict():

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = neigh.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	DeployApp.run(debug=True)