# Visualizations in Python #

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Bring the fields into the visual and adjust the names below --
df = dataset[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})
df['ds'] = pd.to_datetime(df['ds'])

# Aggregate values by different parameters (remove if the aggregation is not wanted) --
# M, A, Q, W and/or mean instead of sum()
monthly_df = df.resample('M', on='ds').sum().reset_index()

model = Prophet()
model.fit(montly_df)

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

//
//

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

//
//

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

//
//

import pandas as pd
import numpy as np

np.random.seed(42)  # for reproducibility

# Configuration
plants = ['P001', 'P002', 'P003']
materials = ['MAT1001', 'MAT1002', 'MAT1003']
num_rows = 200

# Generate random creation dates over ~2 years
date_range = pd.date_range(start='2023-01-01', end='2025-04-30', freq='D')
creation_dates = np.random.choice(date_range, num_rows)

# Simulate plant and material
plant_choices = np.random.choice(plants, num_rows)
material_choices = np.random.choice(materials, num_rows)

# Simulate lead times in days (normal distribution)
# P001: faster, P003: slower
lead_time_means = {'P001': 10, 'P002': 14, 'P003': 18}
lead_times = [
    int(np.clip(np.random.normal(loc=lead_time_means[p], scale=2.5), 5, 30))
    for p in plant_choices
]

# Create the DataFrame
dataset = pd.DataFrame({
    'Plant': plant_choices,
    'Material': material_choices,
    'Creation Date': pd.to_datetime(creation_dates),
    'Leadtime': lead_times
})

# Sort by date for visual consistency
dataset = dataset.sort_values(by='Creation Date').reset_index(drop=True)

# Display a preview
print(dataset.head())

# Save as CSV (optional)
# dataset.to_csv("simulated_leadtime_data.csv", index=False)

