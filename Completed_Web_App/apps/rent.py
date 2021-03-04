import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

def app():

    dummy_df = pd.read_csv('././ListVsRentBedsTotal.csv')

    st.write("""
    # Housing in Metro Atlanta Counties 
    This app predicts the **Rental Price**!
    ### Instructions: Use the sliders on the left to change the value of the features utilized by the model
    * ** List Price: ** The value of the house listing
    * ** Bedrooms: ** number of bedrooms in the house
    * ** County: ** Use the numerical value associated to the county from the table below 
    """)
 

    df = pd.DataFrame({"County Name": ["Fulton", "Gwinnett", "Dekalb", "Cobb", "Other"], "Filter Number": (1,2,3,4,5)})

    # set first td and first th of every table to not display
    st.markdown("""
    <style>
    table td:nth-child(1) {
        display: none
    }
    table th:nth-child(1) {
        display: none
    }
    </style>
    """, unsafe_allow_html=True)

    st.table(df)

    st.write("* Other consists of other counties in the Metro Atlanta Area *")

    st.write('---')

    X = dummy_df[['List Price',
        'Bedrooms','County']]

    y = dummy_df['Rent'].values.reshape(-1,1)

    # Header of Specify Input Parameters
    st.sidebar.header('Specify Input Parameters')

    def user_input_features():
        list_price = st.sidebar.slider('List Price', 90000, 970000, 530000)
        bedrooms = st.sidebar.slider('Bedrooms', 1, 4, 2)
        county1 = st.sidebar.slider('County', 1, 5, 3)
        
        
        data = {'List Price': list_price,
                'Bedrooms': bedrooms,
                'County': county1
                }
        features = pd.DataFrame(data, index=[0])
        return features

    data = user_input_features()

    # Print specified input parameters
    st.header('Specified Input parameters')
    st.write(data)
    st.write('---')

    # Main Panel

    # Build Regression Model
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    from sklearn.preprocessing import StandardScaler
    X_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)

    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    y_train_scaled = y_scaler.transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Apply Model to Make Prediction
    prediction = model.predict(data)

    st.header('Prediction of Potential Monthly Rent')
    st.write(prediction)
    st.write('---')


