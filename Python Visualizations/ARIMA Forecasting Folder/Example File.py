# ARIMA Forecasting #

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset = dataset.sort_values('Date')
dataset.set_index('Date', inplace=True)

# Adjust '24' based on periods to look back on for forecasting --
model = ARIMA(dataset['Sales'], order=(24,1,1)) 
model_fit = model.fit()

forecast = model_fit.forecast(steps=12)

plt.figure(figsize=(10,5))
plt.plot(dataset['Sales'], label='Historical')
plt.plot(forecast, label='Forecast', color='red')
plt.legend()
plt.title('Sales Forecast (ARIMA)')
plt.tight_layout()
plt.show()
