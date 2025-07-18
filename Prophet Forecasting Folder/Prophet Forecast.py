# Visualizations in Python #

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Bring the fields into the visual and adjust the names below --
df = dataset[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})
df['ds'] = pd.to_datetime(df['ds'])

# Aggregate values by different parameters (remove if the aggregation is not wanted) --
# M, A, Q, W and/or mean instead of sum(): Currently Monthly
monthly_df = df.resample('M', on='ds').sum().reset_index()

model = Prophet()
model.fit(monthly_df)

# Adjust the periods as necessary and the date parameter
future = model.make_future_dataframe(periods=12, freq='M')

# Forecast
forecast = model.predict(future)

# Plot forecast
fig = model.plot(forecast)
plt.title("Sales Forecast with Prophet")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()
