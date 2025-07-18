# Utilize holidays #

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.make_holidays import make_holidays_df

df = dataset[['Sale Date', 'Delivery Performance']].rename(columns={'Sale Date': 'ds', 'Delivery Performance': 'y'})
df['ds'] = pd.to_datetime(df['ds'])

monthly_df = df.resample('W', on='ds').mean().reset_index()

# Define known plant holidays --
plant_holidays = pd.DataFrame({
    'holiday': 'plant_closed',
    'ds': pd.to_datetime([
        '2023-12-25', '2024-01-01', '2024-07-04',  # Example US holidays
        '2024-11-28', '2024-12-25', '2025-01-01'   # Add more as needed
    ]),
    'lower_window': 0,
    'upper_window': 0,
})

# Identify Sundays from monthly data (approximate - real closure is daily) --
sundays = df[df['ds'].dt.weekday == 6][['ds']].copy()
sundays['holiday'] = 'sunday_closure'
sundays['lower_window'] = 0
sundays['upper_window'] = 0

custom_holidays = pd.concat([plant_holidays, sundays])

model = Prophet(holidays=custom_holidays)
model.fit(monthly_df)

future = model.make_future_dataframe(periods=12, freq='W')
forecast = model.predict(future)


fig = model.plot(forecast, include_legend=True)
plt.title("Weekly DP Forecast with Prophet")
plt.xlabel("Date")
plt.ylabel("Weekly DP")
plt.tight_layout()
plt.show()
