#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


# In[2]:


Start_date = "2016-01-01"
today= date.today().strftime("%Y-%m-%d")
today


# In[3]:


st.title('Stock App')


# In[4]:


# Define a mapping of company names to their ticker symbols
company_to_symbol = {
    'Google': 'GOOG',
    'Apple': 'AAPL',
    'Microsoft': 'MSFT',
    'GameStop': 'GME'
}

# Let the user select a company
selected_company = st.selectbox('Select company for prediction', list(company_to_symbol.keys()))

# Get the ticker symbol for the selected company
selected_stock = company_to_symbol[selected_company]


# In[17]:


n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


# In[8]:



def load_data(ticker):
    data =yf.download(ticker,Start_date,today)
    data.reset_index(inplace=True)
    return data
data=load_data(selected_stock)


# In[10]:


# Display the raw data
st.subheader('Raw data')
st.write(data.tail())
data.tail()


# In[11]:


def plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name="Stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name="Stock_close"))
    fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()    
        


# In[20]:


df_train =data[['Date','Close']]
df_train =df_train.rename(columns={"Date":"ds","Close":"y"})


# In[15]:


m =Prophet()
m.fit(df_train)


# In[18]:


future=m.make_future_dataframe(periods=period)


# In[19]:


forecast = m.predict(future)


# In[21]:


st.subheader('Forecast data')
st.write(forecast.tail())


# In[23]:


def plot_forecast_data():
    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m,forecast)
    st.plotly_chart(fig1)

plot_forecast_data()    


# In[25]:


st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)


# In[ ]:

# Define a dictionary with stock information
stock_info = {
    'GOOG': {
        'name': 'Alphabet Inc. (Google)',
        'description': 'Alphabet Inc. is a multinational conglomerate that owns Google, YouTube, and more.'
    },
    'AAPL': {
        'name': 'Apple Inc.',
        'description': 'Apple Inc. is an American technology company known for its products like iPhone and Mac.'
    },
    'MSFT': {
        'name': 'Microsoft Corporation',
        'description': 'Microsoft Corporation is a technology company known for its Windows operating system.'
    },
    'GME': {
        'name': 'GameStop Corp.',
        'description': 'GameStop Corp. is a video game and consumer electronics retailer.'
    }
}

# Display stock information
st.subheader('Stock Information')
st.write(f"Name: {stock_info[selected_stock]['name']}")
st.write(f"Symbol: {selected_stock}")
st.write(f"Description: {stock_info[selected_stock]['description']}")

# Provide user instructions
st.subheader('Instructions')
st.write("1. Select a company from the dropdown list to view stock data and forecasts.")
st.write("2. View the forecasted stock prices for the selected number of years.")
st.write("3. Discover the stock's name, symbol, and a brief description.")



