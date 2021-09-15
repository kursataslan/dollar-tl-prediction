import numpy as np
import pandas as pd
import datetime
import math
from fbprophet import Prophet
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

plt.style.use('fivethirtyeight')

import warnings
warnings.filterwarnings('ignore')

veri = pd.read_csv('usdtry.csv')
df=veri.loc[:,["Date","High"]]
df['Date']=pd.DatetimeIndex(df['Date'])
df.dtypes
df=df.rename(columns={'Date': 'ds',
                       'High':'y'
})
ax =df.set_index('ds').plot(figsize=(20,12))
ax.set_ylabel('Değerler')
ax.set_xlabel('Günler')


my_model=Prophet()
my_model.fit(df)

future_dates=my_model.make_future_dataframe(periods=900)
forecast=my_model.predict(future_dates)

fig2=my_model.plot_components(forecast)

forecastnew=forecast['ds']
forecastnew2=forecast['yhat']

forecastnew = pd.concat([forecastnew,forecastnew2], axis=1)

mask = (forecastnew['ds'] > "03-21-2021") & (forecastnew['ds'] <= "09-10-2023")
forecastedvalues = forecastnew.loc[mask]

mask = (forecastnew['ds'] > "02-22-2018") & (forecastnew['ds'] <= "03-21-2021")
forecastnew = forecastnew.loc[mask]

veri1= pd.read_csv('usdtry.csv')
veri1=veri1[['Date','High']]
veri1['Date']=pd.to_datetime(veri1['Date'],format='%m-%d-%Y')
veri1=veri1.sort_values(by=['Date'],ascending=True)
values = veri1['High'].values.reshape(-1,1)
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(values)
TRAIN_SIZE = 0.70
train_size = int(len(dataset) * TRAIN_SIZE)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
def create_dataset(dataset, window_size = 1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i + window_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + window_size, 0])
    return(np.array(data_X), np.array(data_Y))
window_size = 1
train_X, train_Y = create_dataset(train, window_size)
test_X, test_Y = create_dataset(test, window_size)
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
def fit_model(train_X, train_Y, window_size=1):
    model = Sequential()
    model.add(LSTM(100,
                   input_shape=(1, window_size)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",
                  optimizer="adam")
    model.fit(train_X,
              train_Y,
              epochs=5,
              batch_size=1,
              verbose=1)

    return (model)
model1 = fit_model(train_X, train_Y, window_size)

def predict_and_score(model, X, Y):

    pred = scaler.inverse_transform(model.predict(X))
    orig_data = scaler.inverse_transform([Y])
    # kök ortalama kare hatası değeri
    score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))
    return(score, pred)
rmse_train, train_predict = predict_and_score(model1, train_X, train_Y)
rmse_test, test_predict = predict_and_score(model1, test_X, test_Y)
print("Eğitim verisi değer: %.2f RMSE" % rmse_train)
print("Test verisi değer: %.2f RMSE" % rmse_test)

# eğitim veri seti
train_predict_plot = np.empty_like(dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[window_size:len(train_predict) + window_size, :] = train_predict

# test veri seti
test_predict_plot = np.empty_like(dataset)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (window_size * 2) + 1:len(dataset) - 1, :] = test_predict


fig, ax1 = plt.subplots(figsize=(16, 8))
ax1.plot(forecastnew.set_index('ds'),color='b',label="Gerçek Değerler")
ax1.plot(forecastedvalues.set_index('ds'), color='r',label="Tahminler")
ax1.set_ylabel('Değerler')
ax1.set_xlabel('Günler')
plt.legend()
plt.show()

fig, ax2=plt.subplots(figsize=(16,8))
ax2.plot(scaler.inverse_transform(dataset),label="Gerçek Değerler")
ax2.plot(train_predict_plot,label="Eğitim Değerleri")
ax2.plot(test_predict_plot,label="Test Değerleri")
ax2.legend()
plt.show()

