import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 


def app():

    clean_data = pd.read_csv('././dnn_data.csv')

    st.write("""
    # DNN Regression
    This application predicts the **Sale Price**!
    ### Instructions: Use the sliders on the left to change the value of the features utilized by the model
    """)

    st.write('---')

    st.sidebar.header('Specify Input Parameters')

    def user_input_features():
        year_built = st.sidebar.slider('Year Built', 1900, 2010, 1980)
        sq_feet = st.sidebar.slider('Square Feet', 300, 6000, 1464)
        full_bath = st.sidebar.slider('Full Bath', 0, 3, 2)
        half_bath = st.sidebar.slider('Half Bath', 0, 2, 1)
        bedrooms = st.sidebar.slider('Bedrooms', 0, 8, 4)
        total_rooms = st.sidebar.slider('Total Rooms', 2, 14 ,6)
        year_sold = st.sidebar.slider('Year Sold', 2006, 2010, 2008)

        data = {'Year Built': year_built,
                'Square Feet': sq_feet,
                'Full Bath': full_bath,
                'Half Bath' : half_bath,
                'Bedrooms' : bedrooms,
                'Total Rooms' : total_rooms,
                'Year Sold' : year_sold
                }
        features = pd.DataFrame(data, index=[0])
        return features

    data = user_input_features()

    # Print specified input parameters
    st.header('Specified Input parameters')
    st.write(data)
    st.write('---')

    # Main Panel

    # Build DNN Model
    
    X = clean_data.drop(columns= ["SalePrice"])
    y = clean_data[['SalePrice']]   

    model = keras.Sequential()
    model.add(keras.layers.Dense(8, activation = 'relu', input_shape = (7,))) # density based on number of features, relu normalizes the number to 0-1
    model.add(keras.layers.Dense(3, activation = 'relu'))
    model.add(keras.layers.Dense(1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    model.fit(X,y, epochs = 30, callbacks = [keras.callbacks.EarlyStopping(patience = 5)])

    prediction = model.predict(data)

    st.header('Prediction')
    st.write(prediction)
    st.write('---')


    st.header('DNN Creation Process')
    st.write("""
    * ** Data Pre-Processing: ** Split test, train data sets and X, Y labels
    * ** Find Correlation: ** Plot histograms and create a correlation heat map of the features
    * ** Create Deep Neural Network: ** Sequential model, 3 hidden layers, 8 neurons (based on the number of attributes)
    * ** Fitting the model: ** Epochs = 30, Implement callback function 'patience' to avoid overfitting
    """)

    st.write('---')

    st.header('Next Steps')
    st.write("""
    * ** Save as Model: ** Model is currently being re-created everytime user input is given, pickle the model file for quicker use
    * ** Accuracy: ** Compare the accuracy of the model to known labels
    * ** Hyperparameter Tuning: ** Research techniques such as Gradient Descent, Grid Search, Random Search
    """)
        





