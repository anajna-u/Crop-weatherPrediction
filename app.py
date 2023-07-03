from flask import Flask
from flask import request, render_template
from flask_cors import cross_origin
import pickle
import pandas as pd 
import sklearn
import logging 
import numpy as np

app = Flask(__name__)
# model = pickle.load(open("flight_fare_rf.pkl","rb"))
#logging.info('loading the model')


@app.route("/")
@cross_origin()
def home():
    return render_template('home.html', value="shreyas")


@app.route("/predict", methods = ["GET","POST"])
@cross_origin()
def predict():
    

    if request.method == "POST":
        tmp=int(request.form["temp"])
        fert_nit=request.form["nitro"]
        fert_pot=request.form["pot"]
        fert_phos=request.form["phos"]
        rain=request.form["rain"]
        humid=request.form["hum"]
        ph=request.form["ph"]


    prediction = model.predict([[tmp,fert_nit,fert_pot,fert_phos,rain,humid,ph]])
    output= np.argmax(prediction, axis=1)[0]



    return render_template('home.html', prediction_text = "The predicted Area is {}".format(output))

        # return str(request.form)
    
    # if request.method == "POST":
        
    #     fert_nit=request.form["nitro"]
    #     if(fert_nit=='14'):
    #         pass
        # potassium=request.


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
