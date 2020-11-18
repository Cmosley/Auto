import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from collections import namedtuple

## standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scs
import codecs
import math, time

#auto EDA packages
# from pandas_profiling import ProfileReport

#modeling imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#statsmodel imports
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm

from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#sktime models
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import smape_loss
# from sktime.utils.plotting.forecasting import plot_ys 
# from sktime.forecasting.naive import NaiveForecaster
# from sktime.forecasting.arima import AutoARIMA
# from sktime.forecasting.ets import AutoETS
# from sktime.forecasting.compose import EnsembleForecaster

#RNN Model
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, GRU
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# metrics
# from sklearn.metrics import mean_absolute_error as mae
# from sklearn.metrics import accuracy_score

#model export
import pickle

import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

   
# Code Sources: 
# layout and footer inspriation from: https://github.com/Jcharis/Streamlit_DataScience_Apps/blob/master/EDA_app_with_Streamlit_Components/app.py
# modeling inspriation from: https://github.com/Alro10/streamlit-time-series



@st.cache
def load_data1():
    dash = pd.read_csv("./datasets/Dashboard.csv")
    dash['Date'] = pd.to_datetime(dash['Date'])
    dash.set_index('Date', inplace=True)
    return dash

dash = load_data1()

# predictions 
@st.cache
def load_data2():
    preds7 = pd.read_csv("./datasets/7-day-preds.csv")
    preds7.drop(columns=('Unnamed: 0'),inplace=True)
    preds7.rename(columns={"0": "Net Sales"}, inplace=True)
    # dash.set_index('Date', inplace=True)
    return preds7

preds7 = load_data2()

@st.cache
def load_data3():
    preds30 = pd.read_csv("./datasets/30-day-preds.csv")
    preds30.drop(columns=('Unnamed: 0'),inplace=True)
    preds30.rename(columns={"0": "Net Sales"}, inplace=True)
    # dash.set_index('Date', inplace=True)
    return preds30

preds30 = load_data3()

@st.cache
def load_data4():
    preds90 = pd.read_csv("./datasets/90-day-preds.csv")
    preds90.drop(columns=('Unnamed: 0'),inplace=True)
    preds90.rename(columns={"0": "Net Sales"}, inplace=True)
    # dash.set_index('Date', inplace=True)
    return preds90

preds90 = load_data4()

@st.cache
def load_data5():
    df_kpi = pd.read_csv('./datasets/df_kpi.csv')
    df_kpi.drop(columns=('Unnamed: 0'),inplace=True)    
    return df_kpi

df_kpi = load_data5()





footer = """
     <!-- CSS  -->
      <link href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css" type="text/css" rel="stylesheet" media="screen,projection"/>
      <link href="static/css/style.css" type="text/css" rel="stylesheet" media="screen,projection"/>
      <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.5.0/css/all.css" integrity="sha384-B4dIYHKNBt8Bc12p+WXckhzcICo0wtJAoU8YZTY5qE0Id1GSseTk6S+L3BlXeVIU" crossorigin="anonymous">
     
     <footer class="page-footer grey darken-4">
        <div class="container" id="aboutapp">
          <div class="row">
            <div class="col l6 s12">
              <h5 class="white-text">Auto GM</h5>
              <p class="grey-text text-lighten-4">Using my experience in Auto Repair to deliver a metric feedback system for management professionals</p>
            </div>
          
       <div class="col l3 s12">
              
              <ul>
                <a href="https://facebook.com/christian.mosley/" target="_blank" class="light-blue-text">
                <i class="fab fa-facebook fa-6x"></i>
              </a>
              <a href="https://gh.linkedin.com/in/christian-mosley" target="_blank" class="blue-text">
                <i class="fab fa-linkedin fa-6x"></i>
              </a>
               <a href="https://github.com/Cmosley/" target="_blank" class="white-text">
                <i class="fab fa-github-square fa-6x"></i>
              </a>
              </ul>
            </div>
          </div>
        </div>
        <div class="footer-copyright">
          <div class="container">
          Made by <a class="white-text text-lighten-3" href="https://christianmosley.medium.com/">Christian Mosley</a><br/>
          </div>
        </div>
      </footer>
    """



def main():
    """Auto GM - An intelligent business dashboard"""

    menu = ["Home","Dashboard","Forecast","Trends"]
    
    choice = st.sidebar.selectbox("Menu",menu)
    

    # WTD_sales = df_kpi['Net Amount'].iloc[-7:]
    # WTD_sales_num = dash['Net Amount'].iloc[-7:].agg('sum')
    # WTD_cars = dash['Store Ticket Count'].iloc[-7:]
    # WTD_cars_num = dash['Store Ticket Count'].iloc[-7:].agg('sum')

    # #monthly charts
    # MTD_sales = dash['Net Amount'].iloc[-31:]
    # MTD_sales_num = dash['Net Amount'].iloc[-31:].agg('sum')
    # MTD_cars = dash['Store Ticket Count'].iloc[-31:]
    # MTD_cars_num = dash['Store Ticket Count'].iloc[-31:].agg('sum')
    
    # #yearly charts
    # YTD_sales  = dash['Net Amount'].iloc[-316:]
    # YTD_sales_num = dash['Net Amount'].iloc[-316:].agg('sum')
    # YTD_cars = dash['Store Ticket Count'].iloc[-316:]
    # YTD_cars_num = dash['Store Ticket Count'].iloc[-316:].agg('sum')
    	
    

    if choice == 'Dashboard':
        st.subheader("Data Dashboard")
        
        
        kpi_columns_names = df_kpi.columns.tolist()
        
        selected_columns_names = st.selectbox("Select KPI To Plot",kpi_columns_names)
        type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])

        if st.button("Generate Plot"):
            # st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

            # Plot By Streamlit
            if type_of_plot == 'area':
                dash_data = df_kpi[selected_columns_names]
                st.area_chart(dash_data)

            elif type_of_plot == 'bar':
                dash_data = df_kpi[selected_columns_names]
                st.bar_chart(dash_data)

            elif type_of_plot == 'line':
                dash_data = df_kpi[selected_columns_names]
                st.line_chart(dash_data)

            # Custom Plot 
            elif type_of_plot:
                dash_plot= df_kpi[selected_columns_names].plot(kind=type_of_plot)
                st.write(dash_plot)
                st.pyplot()

    # if choice == "Dashboard":
    #         st.subheader(choice)
    #         # drop down for unique value from a column
    #         # time_period = st.selectbox('Select a Time Period', options=["Week(WTD)","Month(MTD)","Year(YTD)"])
    #         # st.line_chart(dash)
    #         st.write()
            
    #         dash_2020 = dash.loc['2020']
            
    #         st.write()

    #         #setup dashboard time periods 
    #         # week_net = alt.Chart(dash.iloc[-7:],width=500,height=400).mark_bar().encode(
    #         #   x="index",
    #         #   y='Net Amount',
    #         #   color="sum(Net Amount)",
    #         #   tooltip="sum(Net Amount)"
    #         # )
    #         # week_cars = alt.Chart(dash.iloc[-7:],width=500,height=400).mark_bar().encode(
    #         #   x="index",
    #         #   y='Store Ticket Count',
    #         #   color="sum(Store Ticket Count)",
    #         #   tooltip="sum(Store Ticket Count)"
    #         # )
    #         # month_net= alt.Chart(dash.iloc[-31:],width=600,height=400).mark_bar().encode(
    #         #   x='Date',
    #         #   y='Net Amount',
    #         #   color="sum(Net Amount)",
    #         #   tooltip="sum(Net Amount)"
    #         # )
    #         # month_cars = alt.Chart(dash.iloc[-31:],width=600,height=400).mark_bar().encode(
    #         #   x='Date',
    #         #   y='Store Ticket Count',
    #         #   color="sum(Store Ticket Count)",
    #         #   tooltip="sum(Store Ticket Count)"
    #         # )
            
    #         # year_net= alt.Chart(dash_2020,width=800,height=400).mark_line().encode(
    #         #   x='Date',
    #         #   y='Net Amount',
    #         #   color="sum(Net Amount)",
    #         #   tooltip="sum(Net Amount)"
    #         # )
            
    #         # year_cars = alt.Chart(dash_2020,width=800,height=400).mark_line().encode(
    #         #   x='Date',
    #         #   y='Store Ticket Count',
    #         #   color="sum(Store Ticket Count)",
    #         #   tooltip="sum(Store Ticket Count)"
    #         # )
    #         # weekly charts
    #         WTD_sales = dash['Net Amount'].iloc[-7:]
    #         WTD_sales_num = dash['Net Amount'].iloc[-7:].agg('sum')
    #         WTD_cars = dash['Store Ticket Count'].iloc[-7:]
    #         WTD_cars_num = dash['Store Ticket Count'].iloc[-7:].agg('sum')

    #         #monthly charts
    #         MTD_sales = dash['Net Amount'].iloc[-31:]
    #         MTD_sales_num = dash['Net Amount'].iloc[-31:].agg('sum')
    #         MTD_cars = dash['Store Ticket Count'].iloc[-31:]
    #         MTD_cars_num = dash['Store Ticket Count'].iloc[-31:].agg('sum')
            
    #         #yearly charts
    #         YTD_sales  = dash['Net Amount'].iloc[-316:]
    #         YTD_sales_num = dash['Net Amount'].iloc[-316:].agg('sum')
    #         YTD_cars = dash['Store Ticket Count'].iloc[-316:]
    #         YTD_cars_num = dash['Store Ticket Count'].iloc[-316:].agg('sum')
            
    #         # radio button selectors 
    #         dash_display = st.radio("What time period?", ("Week", "Month", "Year"), key="dash")
            
    #         if dash_display == "Week":
    #             st.line_chart(WTD_sales, width=500, height=250)
    #             st.write('Total Sales for the Week of Oct 13th:', WTD_sales_num)
    #             st.line_chart(WTD_cars, width=500, height=250)
    #             st.write('Total Cars for the Week of Oct 13th:', WTD_cars_num)
    #             # st.altair_chart(week_net)
    #             # st.altair_chart(week_cars)
    #         elif dash_display == "Month": 
    #             st.line_chart(MTD_sales, width=500, height=250)
    #             st.write('Total Sales for the the last month:', MTD_sales_num)
    #             st.line_chart(MTD_cars, width=500, height=250)
    #             st.write('Total Cars for the the last month:', MTD_cars_num)
    #             # st.altair_chart(month_net)
    #             # st.altair_chart(month_cars)
    #         else: 
    #             st.line_chart(YTD_sales, width=500, height=250)
    #             st.write('Total Sales for the year:', YTD_sales_num)
    #             st.line_chart(YTD_cars, width=500, height=250)
    #             st.write('Total Cars for the year:', YTD_cars_num)
    #             # st.altair_chart(year_net)
                
    #             # st.altair_chart(year_cars)
    
         
            

    elif choice == "Forecast":
        st.subheader("Forecasts")
        
        # load models           
        # holt_winter = open("hw_preds.pkl", "rb")
        # hw = pickle.load(holt_winter)    
        
        # with open("hw_preds.pkl", "rb") as hw:
        #   holt = pickle.load(hw)
        
        # sarima_model = open("sarima_model.pkl", "rb")
        # sarima = pickle.load(sarima_model) 
        
        # with open("sarima_model.pkl", "rb") as sm:
        #   sarima = pickle.load(sm)
        
        # naive_model_mean = open("naive_model_mean.pkl", "rb")
        # naive_mean = pickle.load(naive_model_mean)
        # with open("naive_model_mean.pkl", "rb") as nmm:
        #   mean = pickle.load(nmm)
        
        st.write('Choose the time period to predict')
        forecast_period = st.selectbox("Time Period:", ["7 Day", "30 Days", "90 Days"])
        if st.button("Generate Plot"):
          if forecast_period == "7 Day": 
            st.markdown("**7 Day Forecast**")
            st.line_chart(preds7)
            st.markdown("**Predictions**")
            st.write(preds7)
          elif forecast_period == "30 Days": 
            st.markdown("**30 Day Forecast**")
            st.line_chart(preds30)
            st.markdown("**Predictions**")
            st.write(preds30)
          else: 
            st.markdown("**90 Day Forecast**")
            st.line_chart(preds90)
            st.markdown("**Predictions**")
            st.write(preds90)
            
            
        # st.button("Predict")
        #   # model_selection = st.selectbox("Model to train", ["AutoArima", "LSTM", "MLP", "RNN"])
        #       # if model_selection == "AutoArima":
        # prediction = sarima.predict(np.arange(6 + 1)
        
        
        # make predictions
        # prediction = lr.predict(user_input.reshape(1, -1))
        # st.header(f"The model predicts: ${np.round(prediction[0])}")
        


    elif choice == "Trends":
        st.subheader("Trends")
        st.write("UNDER CONSTRUCTION")
        

    else:
        image = Image.open('./assets/Auto-gm-logo.png')
        st.image(image, use_column_width=True)
        components.html(footer, height=500)



        html_temp = """
        <div style="background-color:royalblue;padding:10px;border-radius:10px">
        <h1 style="color:white;text-align:center;">Simple EDA App with Streamlit Components</h1>
        </div>
        """


if __name__ == '__main__':
    main()