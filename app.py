# FLask API is a tool which helps to connect webs servers to your project

from flask import Flask, render_template,url_for,request,jsonify
import os
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('home.html')

@app.route('/result',methods=['POST','GET'])
def result():
    cylinders=int(request.form["cylinders"])
    displacement=int(request.form["displacement"])
    horsepower=int(request.form["horsepower"])
    weight=int(request.form["weight"])
    acceleration=int(request.form["acceleration"])
    model_year=int(request.form["model_year"])
    origin=int(request.form["origin"])
    
    values=[[cylinders,displacement,horsepower,weight,acceleration,model_year,origin]]

    sc = pickle.load(open('scaler.pkl','rb'))
        
    values=sc.transform(values)

    model=load_model("model.h5")

    prediction=model.predict(values)
    prediction=float(prediction)
    print(prediction)
    
    json_dict={
        "prediction":prediction
    }   
    
    return jsonify(json_dict)
    
if __name__=="__main__":
    app.run(debug=True)
