import streamlit as st
from multiapp import MultiApp
from apps import home, rent, data, kim, greg, grant, about

app = MultiApp()

st.markdown("""
This application is powered by Streamlit, Pandas, Sklearn, & Tableau
""")

app.add_app("Home", home.app)
app.add_app("Rental Price Predictor", rent.app)
app.add_app("Statistical Insights", data.app)
app.add_app("Crime vs Listing Price", kim.app)
app.add_app("Race vs Listing Price", grant.app)
app.add_app("Turner & Suntrust Impact", greg.app)
app.add_app("About Us", about.app)

app.run()