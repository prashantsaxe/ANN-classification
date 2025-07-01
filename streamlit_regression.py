import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder

model = tf.keras.models.load_model('regression_model.h5')


with open('onehot_encoder_geo_regression.pkl','rb') as file:
    onehot_encoder_geo_regression = pickle.load(file)
with open('label_encoder_gender_regression.pkl','rb') as file:
    label_encoder_gender_regression = pickle.load(file)
with open('scaler_regression.pkl','rb') as file:
    scaler_regression = pickle.load(file)

st.title("customer's estimated salary prediction ")

geography = st.selectbox('Geography',onehot_encoder_geo_regression.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender_regression.classes_)
age = st.slider('Age',18 , 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure',0,10)
number_of_products = st.slider('Number of Product',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])
Exited = st.selectbox('Exited',[0,1])


input_data =pd.DataFrame( {
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender_regression.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited' : [Exited]
})

geo_encoded = onehot_encoder_geo_regression.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo_regression.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_data_sclaed = scaler_regression.transform(input_data)

prediction = model.predict(input_data_sclaed)
prediction_salary = prediction[0][0]

st.write(f'predicted salary is : {prediction_salary:.2f}')
