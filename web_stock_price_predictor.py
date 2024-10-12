import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf

st.title("Stock Price Predictor App")

stock = st.text_input("Enter the stock ID", "GOOG")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

df = yf.download(stock, start, end)

model = load_model("Stock_Price_model.keras")
st.subheader("Stock Data")
st.write(df)

splitting_len = int(len(df)*0.7)
x_test = pd.DataFrame(df.Close[splitting_len:])

def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize = figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close Price and MA for 250 days')
df['MA_for_250_days'] = df.Close.rolling(250).mean()
st.pyplot(plot_graph((15, 6), df['MA_for_250_days'], df, 0))

st.subheader('Original Close Price and MA for 200 days')
df['MA_for_200_days'] = df.Close.rolling(200).mean()
st.pyplot(plot_graph((15, 6), df['MA_for_200_days'], df, 0))

st.subheader('Original Close Price and MA for 100 days')
df['MA_for_100_days'] = df.Close.rolling(100).mean()
st.pyplot(plot_graph((15, 6), df['MA_for_100_days'], df, 0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
df['MA_for_100_days'] = df.Close.rolling(100).mean()
st.pyplot(plot_graph((15, 6), df['MA_for_100_days'], df, 1, df['MA_for_250_days']))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100: i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

plotting_data = pd.DataFrame(
    {
        'Original_test_data':inv_y_test.reshape(-1),
        'Predictions': inv_pre.reshape(-1)
    },
        index = df.index[splitting_len+100:]
)
st.subheader("Original values vs Predicted Values")
st.write(plotting_data)

st.subheader("Original close price vs Predicted Close price")
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([df.Close[:splitting_len+100], plotting_data], axis=0))
plt.legend(["Data Not Used", "Original Test data", "Predicted Test Data"])
st.pyplot(fig)