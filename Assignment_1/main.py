import pandas as pd
import numpy as np

df = pd.read_excel("AirQualityUCI.xlsx")

df['datetime'] = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.drop(['Date', 'Time'], axis=1)
df = df.set_index('datetime')
df = df.sort_index()
# print(df.head())

df.replace(-200, np.nan, inplace=True)
# print(df.isnull().sum())

df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)
# print(df.isnull().sum())
# print(df.head())

# Create time-based features
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = df.index.weekday >= 5  # 1 for Saturday/Sunday, 0 for weekdays

# Define target columns
target_cols = df.columns.tolist()[:-4]
# Create target columns by shifting the pollutants by 1 step
df[target_cols] = df[target_cols].shift(-1)
# Drop the last row
df.dropna(inplace=True)


# Split data into features (X) and targets (y)
X = df.drop(target_cols, axis=1)  # Features (pollutants, time-based features)
y = df[target_cols]  # Targets (next values of pollutants)

# Split the data into training and testing sets (leave out the last 48 hours)
X_train = X[:-48]
y_train = y[:-48]

X_test = X[-48:]
y_test = y[-48:]

# print("X_train shape:", X_train.shape)
# print("y_train shape:", y_train.shape)
# print(X_train.head())
# print(y_train.head())

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

# Initialize the XGBoost model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)

# Wrap the model with MultiOutputRegressor to handle multiple pollutants
multi_output_model = MultiOutputRegressor(xgb_model)

# Train the model
multi_output_model.fit(X_train, y_train)

# Make predictions on the test set (next 48 hours)
y_pred = multi_output_model.predict(X_test)

# Convert predictions into a DataFrame for easier reading
y_pred_df = pd.DataFrame(y_pred, columns=target_cols)
y_pred_df.index = pd.date_range(start=df.index[-48] + pd.Timedelta(hours=1), periods=48, freq='H')



from sklearn.metrics import mean_squared_error

for col in target_cols:
    rmse = np.sqrt(mean_squared_error(y_test[col], y_pred_df[col]))
    print("RMSE:", rmse, "for", col)







print(target_cols)
y_pred_df['Date'] = y_pred_df.index.date
y_pred_df['Time'] = y_pred_df.index.time

# Reorder columns: move 'date' and 'time' to the beginning
cols = ['Date', 'Time'] + [col for col in y_pred_df.columns if col not in ['Date', 'Time']]
y_pred_df = y_pred_df[cols]


# Remove the datetime index and drop it
final = y_pred_df.reset_index(drop=True)
print(final.head())
# Save DataFrame to a new Excel file
final.to_excel('submissions.xlsx', index=False)
