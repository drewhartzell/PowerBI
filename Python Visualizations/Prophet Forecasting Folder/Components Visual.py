# Components #

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

df = dataset[['Sale Date', 'Delivery Performance']].rename(columns={'Sale Date': 'ds', 'Delivery Performance': 'y'})
df['ds'] = pd.to_datetime(df['ds'])

# Aggregate daily data into monthly revenue (summing values for each month) --
monthly_df = df.resample('M', on='ds').mean().reset_index()

model = Prophet()
model.fit(monthly_df)

# Forecast future periods (12 months into the future) --
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

# Plot forecast
fig2 = model.plot_components(forecast)
plt.title("Monthly DP Forecast with Prophet")
plt.xlabel("Date")
plt.ylabel("Monthly DP")
plt.tight_layout()
plt.show()
