from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing

app = Flask(__name__)
pickle_in=open('telecom_model.pkl', 'rb')
model = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Telecom Churn prediction model"

@app.route('/predict')
def prediction():
    tenure = request.args.get('tenure')
    PhoneService = request.args.get('PhoneService')
    PaperlessBilling = request.args.get('PaperlessBilling')
    MonthlyCharges = request.args.get('MonthlyCharges')
    TotalCharges = request.args.get('TotalCharges')
    SeniorCitizen = request.args.get('SeniorCitizen')
    Partner = request.args.get('Partner')
    Dependents = request.args.get('Dependents')
    Contract_One_year = request.args.get('Contract_One year')
    Contract_Two_year = request.args.get('Contract_Two year')
    PaymentMethod_Credit_card_automatic = request.args.get('PaymentMethod_Credit card (automatic)')
    PaymentMethod_Electronic_check = request.args.get('PaymentMethod_Electronic check')
    PaymentMethod_Mailed_check = request.args.get('PaymentMethod_Mailed check')
    InternetService_Fiber_optic = request.args.get('InternetService_Fiber optic')
    InternetService_No = request.args.get('InternetService_No')
    OnlineSecurity_Yes = request.args.get('OnlineSecurity_Yes')
    OnlineBackup_Yes = request.args.get('OnlineBackup_Yes')
    DeviceProtection_Yes = request.args.get('DeviceProtection_Yes')
    TechSupport_Yes = request.args.get('TechSupport_Yes')
    col_names = ['tenure', 'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Partner', 'Dependents', 'Contract_One year', 'Contract_Two year', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes', 'TechSupport_Yes']
    X_values = pd.DataFrame([tenure, PhoneService, PaperlessBilling, MonthlyCharges, TotalCharges, SeniorCitizen, Partner, Dependents, Contract_One_year, Contract_Two_year, PaymentMethod_Credit_card_automatic, PaymentMethod_Electronic_check, PaymentMethod_Mailed_check, InternetService_Fiber_optic, InternetService_No, OnlineSecurity_Yes, OnlineBackup_Yes, DeviceProtection_Yes, TechSupport_Yes], index=col_names).T
    X_values = X_values.apply(pd.to_numeric)
    prediction = model.predict(X_values)
    if str(prediction) == '[1]':
        return "The customer will churn"
    elif str(prediction) == '[0]':
        return "The customer will stay"
    else:
        return "Error"
    

if __name__ == '__main':
    app.run()