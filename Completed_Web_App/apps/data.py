import streamlit as st
import pandas as pd

def app():

    data = pd.read_csv('././ListVsRentBedsTotal.csv')

    st.write("""
    # Rental Price Data
    """)
    st.write("""When assessing the Zillow data from Quandl, we wanted to see if anything had a significant impact on rent prices. We looked at these three things:
    * The number of days a property is on Zillow and the effect that has on the rent price
    * Square footage of a property and the effect that has on rent price
    * Number of bedrooms a property has and the effect that has on rent price""")

    st.write('---')

    st.write('**Days on Zillow**')
    st.write('When looking at the number of days on Zillow and rent prices, the plot below shows there is not much of a correlation between the two. The r-squared is: 0.2424063747309388')
    st.image("././R-squared-DaysOnZillow.png")

    st.write("**Square Footage**")
    st.write('When looking at the square footage of a property and rent prices, the plot below shows there is some correlation between the two. The r-squared is: 0.7967550272920328')
    st.image("././R-squared-Sqft.png")

    st.write("""Looking deeper at the square footage data, we made these other plots comparing List Price per Sqft, Rent price per Sqft and Sold price per Sqft: Looking at the bottom row of plots in the Pairplot below, you can see that there is definite correlation between the Median Rent Price per Sqft and Median Rent prices. Some between Median List Price Price per Sqft and Median Rent Prices. And little between Median Sold Price per Sqft and Median Rent Price. """)
    st.image("././PairPlot-Sqft.png")

    st.write("""In this 3D plot, you can see the same. A definite correlation between the Median Rent Price per Sqft and Median Rent prices. Some between Median List Price Price per Sqft and Median Rent Prices. And little between Median Sold Price per Sqft and Median Rent Price. """)
    st.image("././3DPlot-Sqft.png")

    st.write("""This residual plot shows MSE: 0.3465646443506312 and R2: 0.6114347993692942""")
    st.image("././residual_plotq.png")

    st.write("**Number of Bedrooms**")
    st.write("""When looking at the number of bedrooms a property has and rent prices, the bottom row of plots in the Pairplot below shows there is definite correlation between the two.""")
    st.image("././PairPlot-#BR.png")

    st.write("""The plot below shows the high r-squared of: 0.9623828220461137 between Rent Prices and Median Rental Price for 3 bedroom rentals.""")
    st.image("././R-squared-#BR.png")

    st.write("""The 3D plot below displays the comparison of Median Rental Price-1BR, Median Rental Price- 3BR and Median Rent Price. You can see there is a very high correlation. """)
    st.image("././3DPlot-#BR.png")

    st.write("**Conclusion**")
    st.write("""we see that the number of bedrooms a rental has plays a large part in rental prices. The square footage has a small impact, and the days the rental is listed on Zillow has a very small impact.""")
    



    


