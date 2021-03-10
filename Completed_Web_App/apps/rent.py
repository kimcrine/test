import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def app():

    dummy_df = pd.read_csv('././ListVsRentBedsTotal.csv')

    st.write("""
    # Housing in Metro Atlanta Counties 
    This app predicts the **Rental Price**!
    ### Instructions: Use the sliders on the left to change the value of the features utilized by the model
    * ** List Price: ** The value of the house listing
    * ** Bedrooms: ** number of bedrooms in the house
    * ** County: ** Select county you would like to specify
    """)

    st.write('---')

    # Header of Specify Input Parameters
    st.sidebar.header('Specify Input Parameters')

    def user_input_features():
        list_price = st.sidebar.slider('List Price', 90000, 970000, 530000)
        bedrooms = st.sidebar.slider('Bedrooms', 1, 4, 2)
        county_name = st.sidebar.selectbox("Counties", ("Fulton", "Gwinnett", "Dekalb", "Cobb", "Other"))

        def decode(county):
            dictionary = {1: "Fulton", 2: "Gwinnett", 3: "Dekalb", 4: "Cobb", 5: "Other"}
            county_str = ""
            for key, value in dictionary.items():
                if county == value:
                    county_str = key
            return county_str

        county1 = decode(county_name)
        
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

    X = dummy_df[['List Price',
        'Bedrooms','County']]

    y = dummy_df['Rent'].values.reshape(-1,1)

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

    if st.button("Click to see more information on the model"):
        st.write("""We selected the **Linear Regression Model** because we wanted to make a useful tool for calculating possible rental income that took into account the initial cost of the property (in addition to location). Due to the linear relationship between purchase price and rent, we decided a Linear Regression would be an effective tool for predictions.""")
        st.write("""The visualization below plots the difference between the model predicted values and actual y values, versus the model predicted values
        """)
        st.write('---')
        predictions = model.predict(X_test_scaled)
        model.fit(X_train_scaled, y_train_scaled)
        fig = plt.figure()
        plt.scatter(model.predict(X_train_scaled), model.predict(X_train_scaled) - y_train_scaled, c="blue", label="Training Data")
        plt.scatter(model.predict(X_test_scaled), model.predict(X_test_scaled) - y_test_scaled, c="orange", label="Testing Data")
        plt.legend()
        plt.hlines(y=0, xmin=y_test_scaled.min(), xmax=y_test_scaled.max())
        plt.title("Residual Plot")

        st.pyplot(fig)


        from sklearn.metrics import mean_squared_error

        MSE = mean_squared_error(y_test_scaled, predictions)
        r2 = model.score(X_test_scaled, y_test_scaled)

        st.write("MSE:", round((MSE),2), "R2:", round((r2),2))





