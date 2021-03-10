import streamlit as st

def app():
    st.title('Home')

    st.write('Welcome to our `Housing Predictor` application.')

    st.write('In this app, we will be building a simple machine learning model using the Metro Atlanta dataset.')

    st.write('---')

    st.write("""
     **Introduction**

    For our final project, we decided to do an analysis and run machine learning models on Atlanta real estate data. The data we used was the Zillow Real Estate Data on Quandl.com. We ran API calls to pull all available Zillow real estate data for all metro Atlanta zip codes. We created an app where users can enter the listing price of a house, the number of bedrooms, and the zip code the house is located in, and it would determine and show the estimated monthly rental price of the house.
    """)

    st.write("""
     **Cleaning Process**

    We then proceeded to clean the data using Python within Jupyter Notebooks. Cleaning involved removing rows with null values, changing column names, and merging the Zillow data with other datasets including Atlanta census data and Atlanta crime data.
    """)

    st.write("""
     **Questions Asked**
    After cleaning, we analyzed the data to answer various questions. The questions we attempted to answer and visualize were:
    * What are the median sales prices and median rental prices for each county, city, and zip code in the Atlanta metro area? Which are the highest? Which are the lowest?
    * What is the correlation between the percentages of minorities in a zip code and home sales prices and rental prices?
    * What is the correlation between crime in a zip code and home sales prices and rental prices?
    * What effect does an event like Turner Field closing and SunTrust Park opening have on home prices in those areas?
    * How do things like square footage and number of bedrooms correlate with the rental prices?

    """)

    st.write("""
    **Machine Learning Model**

    For our project we decided to use Linear Regression.
    """)

    st.write("""
    **Findings**
    * We used Tableau to visualize median house sales prices and rental prices for each zip code in the Atlanta metro area. We found that the city is basically split in half in terms of housing prices with most of the northern metro having above-average house sales and rental prices and most of the southern metro having below-average house sales and rental prices. The county with the highest median sales price is Fayette county at $266,663. The county with the lowest median sales price is Clayton county at $74,763. The cities with the highest median sales prices are Alpharetta, Roswell, and Peachtree City at $345,890, $335,513, and $325,960 respectively. The cities with the lowest median sales prices are all in Clayton county - Riverdale, Forest Park, and Conley at $72,858, $47,050, and $27,000 respectively. The zip code with by far the highest median sales price is 30327 which covers West Buckhead at $709,450. The zip code with the lowest median sales price is 30288 which covers the city of Conley at $27,000. 
    * We found that there is a strong negative correlation between the percentage of minorities in a given zip code and the median sales prices and rental prices of houses in the zip code, meaning the higher the percentage of minorities in the zip code, the lower the median sales price or rental price of houses in the zip code on average. In addition, we found that zip codes that are majority minorities median sales price was over $110,000 less than zip codes that are majority whites - $151,775 majority minorities vs. $264,000 for majority whites. Similarly, zip codes that are majority minorities have median rental prices much lower on average than zip codes that are majority white - $1,395 and $1,800 respectively.
    * We found that number of bedrooms strongly correlated with the Rent Price. Square footage had a small correlation, and Days on Zillow did not correlate very much at all with the rent price. 
    """)

    st.write("""
    **Conclusions & Challenges**

    * As much data as Zillow has, when you zoom in past region, it shows gaps.  Some features don’t return any results when searched by zip code, like foreclosures per 10,000 homes, or days on ZIllow.  Other features are present for some zip codes but not others, like number of monthly listings.  When viewing features by zip code and date, some return lots of relevant data (2003 - 2020), while other zip codes only return one year (2018-2019) or much older dates (1984-1997).
    * The Zillow data was only available on a county or zip code level, so trying to combine that with data on a latitude/longitude level became a challenge. We were able to work around this using proxy map layers for now.

    """)

