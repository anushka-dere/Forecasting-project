import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
import datetime
import base64

st.title('Time Series Forecasting Using Streamlit')


appdata = pd.read_csv(r"C:\Users\Admin\Documents\excelr\project\forecasting\datasetnew2.csv")
appdata['ds'] = pd.to_datetime(appdata['ds'],errors='coerce') 
    
    
max_date = appdata['ds'].max()

st.write("SELECT FORECAST PERIOD")

periods_input = st.number_input('How many days forecast do you want?',min_value = 0, max_value = 3650)
obj = Prophet()
obj.fit(appdata)

future = obj.make_future_dataframe(periods=periods_input)

   
fcst = obj.predict(future)


if st.button("Forecast"):
 st.write("The following table shows future predicted values. 'yhat' is the predicted value; upper and lower limits are 80% confidence intervals by default")    
 forecast = fcst[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
 
 forecast_filtered =  forecast[forecast['ds'] > max_date]    
 st.write(forecast_filtered)

 st.write("VISUALIZE FORECASTED DATA")
    
 st.write("The next visual shows the actual (black dots) and predicted (blue line) values over time.")    

 figure1 = obj.plot(fcst)
 st.write(figure1)
 
    
 st.write("The next few visuals show a high level trend of predicted values, day of week trends, and yearly trends (if dataset covers multiple years). The blue shaded area represents upper and lower confidence intervals.")
      

 figure2 = obj.plot_components(fcst)
 st.write(figure2)