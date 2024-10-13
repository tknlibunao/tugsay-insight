import streamlit as st
import altair as alt
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import xgboost as xgb
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import keras_tuner as kt
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import KFold
import datetime as dt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

### STREAMLIT APPLICATION
st.set_page_config(page_title="INSIGHT: Integrated Naïve Forecasting Systems for Interstitial Glucose Forecasting—Hybridized with Machine and Deep Learning Techniques", page_icon="📈")

### DEFINE FUNCTIONS, SESSION STATES, AND CONSTANTS
def get_food_data():
    # Load the CSV file (food data)
    food_df = pd.read_csv('food_data.csv')

    # Create a dictionary where food names are the keys (labels) and the carbohydrate amounts are the values
    food_dict = dict(zip(food_df['Food'], food_df['Amount of carbohydrates, grams per 100 grams']))
    return food_dict

# FOOD DICTIONARY
food_dict = get_food_data()

if 'is_running' not in st.session_state:
    st.session_state.is_running = False

if 'user_input_status' not in st.session_state:
    st.session_state.user_input_status = "USER INPUT STATUS: (no user input yet)"

if 'selected_food' not in st.session_state:
    st.session_state.selected_food = sorted(list(food_dict.keys()))[0]

# Store the food amount in session state to persist the input
if 'amount_in_grams' not in st.session_state:
    st.session_state.amount_in_grams = 0.0

if 'carbohydrates' not in st.session_state:
    st.session_state.carbohydrates = 0.0

# Initialize session states for data point counter and cumulative data
if 'data_point_counter' not in st.session_state:
    st.session_state.data_point_counter = 0

if 'cumulative_added_data' not in st.session_state:
    st.session_state.cumulative_added_data = pd.DataFrame()

## USER INPUT
# Update the session state with the selected forecast horizon for LSTM
def set_selected_forecast_horizon_index(n):
    st.session_state.selected_forecast_horizon_index = n

# Forecast horizons for the LSTM model
forecast_horizons_list = [
    {"label": "5 MINS", "value": 1},
    {"label": "15 MINS", "value": 3},
    {"label": "30 MINS", "value": 6},
    {"label": "1 HOUR", "value": 12},
    {"label": "3 HOURS", "value": 36},
    {"label": "6 HOURS", "value": 72},
    {"label": "12 HOURS", "value": 144},
    {"label": "24 HOURS", "value": 288}
]

# Make "24 HOURS" the default selected forecast horizon
default_forecast_horizon_index = 7

## NAIVE 2
# Remove outliers using median
def remove_outliers_with_median(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    median = series.median()
    return np.where((series < lower_bound) | (series > upper_bound), median, series)

# Calculate the exponential moving average for Naive 2
def calculate_ema(series, span=10):
    return series.ewm(span=span, adjust=False).mean()

## XGBOOST
# Remove outliers using median
def remove_outliers_and_replace_with_median(data, threshold=2):
    if isinstance(data, pd.DataFrame):
        data_array = data.values
        columns = data.columns
    else:
        data_array = data
    
    # Calculate z-scores to detect outliers
    z_scores = np.abs(stats.zscore(data_array, axis=0))
    mask = z_scores < threshold 

    cleaned_data = np.where(mask, data_array, np.nan)
    
    for i in range(cleaned_data.shape[1]):
        if np.isnan(cleaned_data[:, i]).all():
            interval_median = 0 
        else:
            interval_median = np.nanmedian(cleaned_data[:, i])
        cleaned_data[:, i] = np.where(np.isnan(cleaned_data[:, i]), interval_median, cleaned_data[:, i])
    
    if isinstance(data, pd.DataFrame):
        cleaned_data = pd.DataFrame(cleaned_data, columns=columns, index=data.index)
    
    return cleaned_data

# Sugar decay function with exponential moving average
def sugar_decay(sugar_series, decay_rate=0.1):
    return sugar_series.ewm(span=10, adjust=False).mean() * decay_rate

## LSTM
# Define past residuals
time_step = 230

# Create dataset
def create_dataset(data, time_step=time_step):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Add user input (food intake)
def add_sugar_intake(user_sugar_intake):
    st.session_state.user_input_status = "USER INPUT STATUS: (has added some user sugar intake)"
    
    # Create a DataFrame for the new row
    new_row = pd.DataFrame({
        'Date': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:00')],
        'SugarIntake': [user_sugar_intake]
    })

    # Append the new row to the sugar_df in session state
    st.session_state.sugar_df = pd.concat([st.session_state.sugar_df, new_row], ignore_index=True)

    # Sort the sugar_df by Date to ensure it's monotonic
    st.session_state.sugar_df['Date'] = pd.to_datetime(st.session_state.sugar_df['Date'])
    st.session_state.sugar_df = st.session_state.sugar_df.sort_values(by='Date').reset_index(drop=True)

    st.success(f"Added {user_sugar_intake:.2f} grams of carbohydrates from {st.session_state.selected_food}. Will be used in the next cycle.")

# Function to simulate glucose data updates every X minutes with forecasting
def simulate_data_addition_with_forecasting(df, original_sugar_df, last_hour_data, interval_minutes=5):
    if st.session_state.is_running:
        start_time = dt.datetime.now()  # Track start time of the simulation
        run_duration_minutes = 5  # Total simulation time (1 hour) -- CHANGE TO 60 if 1 hour, SET to 5 for demo
        elapsed_time = 0

        data_point_counter = st.session_state.data_point_counter  # Keep track of how many data points have been added
        cumulative_added_data = st.session_state.cumulative_added_data  # Keep track of cumulative added data

        user_input_status_placeholder = st.empty() # Placeholder for testing
        cycle_counter_placeholder = st.empty() # Placeholder for testing counter

        table_title_placeholder = st.empty()  # Placeholder for the table title
        table_placeholder = st.empty()  # Placeholder for the table

        sugar_table_title_placeholder = st.empty()  # Placeholder for the sugar table title
        sugar_table_placeholder = st.empty()  # Placeholder for the sugar table

        while elapsed_time < run_duration_minutes and st.session_state.is_running:
            user_input_status_placeholder.write(st.session_state.user_input_status)
            cycle_counter_placeholder.write(f"CYCLE COUNTER: {st.session_state.data_point_counter + 1}")


            # Add one data point from the last hour every 5 minutes
            if data_point_counter < len(last_hour_data):
                new_data_point = last_hour_data.iloc[[data_point_counter]]
                df = pd.concat([df, last_hour_data.iloc[[data_point_counter]]])
                cumulative_added_data = pd.concat([cumulative_added_data, new_data_point])

                # Display cumulative added data for checking
                table_title_placeholder.write("Cumulative Added Data (so far):")
                table_placeholder.write(cumulative_added_data)

                # Display existing sugar_df for reference
                sugar_table_title_placeholder.write("Current Sugar DataFrame:")
                sugar_table_placeholder.write(st.session_state.sugar_df)

            # # Make a copy of sugar_df to avoid in-place modifications
            current_sugar_df= st.session_state.sugar_df.copy()

            # CELL 2
            # CLEAN THE GLUCOSE DATA
            df_cleaned = df.dropna(subset=['Interstitial Glucose Value'])

            # MEAN AND STANDARD DEVIATION OF IGL
            mean_igv = df_cleaned['Interstitial Glucose Value'].mean()
            std_igv = df_cleaned['Interstitial Glucose Value'].std()
            print(f"Mean of Interstitial Glucose Value: {mean_igv:.2f} mg/dL")
            print(f"Standard Deviation of Interstitial Glucose Value: {std_igv:.2f} mg/dL")

            # CELL 3
            # THIS CODE IS FOR THE NAIVE 1 MODEL
            # TEST SET
            future_index = pd.date_range(df_cleaned.index[-1], periods=288 + 1, freq='5min')[1:]  
            test_data = pd.DataFrame(index=future_index)

            # TRAIN SET
            train_data = df_cleaned.copy() 

            # EXTRACT THE TIME OF DAY FROM THE INDEX FOR GROUPING
            train_data.loc[:, 'time_of_day'] = train_data.index.time
            test_data.loc[:, 'time_of_day'] = test_data.index.time

            # LOOP
            average_forecasts = {}

            for time_interval in test_data['time_of_day'].unique():
                values_at_time = train_data[train_data['time_of_day'] == time_interval]['Interstitial Glucose Value'].dropna()
                
                # CALCULATE THE AVERAGE AND IGNORE MISSING VALUES
                if len(values_at_time) > 0:
                    average_forecasts[time_interval] = values_at_time.mean()
                else:
                    average_forecasts[time_interval] = np.nan 

            # MAP THE FORECASTED VALUES TO THE TEST SET BASED ON TIME OF THE DAY
            test_data.loc[:, 'forecast'] = test_data['time_of_day'].map(average_forecasts)

            # CELL 4

            # TEST SET for NAIVE 2
            future_index = pd.date_range(df_cleaned.index[-1], periods=288 + 1, freq='5min')[1:]  
            test_data_2 = pd.DataFrame(index=future_index)

            # TRAIN SET for NAIVE 2
            train_data_2 = df_cleaned.copy()

            # EXTRACT THE TIME OF DAY FROM THE INDEX FOR GROUPING
            train_data_2.loc[:, 'time_of_day'] = train_data_2.index.time
            test_data_2.loc[:, 'time_of_day'] = test_data_2.index.time

            # COMPUTE FORECAST FOR NAIVE 2
            average_forecasts_2 = {}
            for time_interval in test_data_2['time_of_day'].unique():
                values_at_time_2 = train_data_2[train_data_2['time_of_day'] == time_interval]['Interstitial Glucose Value'].dropna()
                values_at_time_2 = pd.Series(remove_outliers_with_median(values_at_time_2))
                if len(values_at_time_2) > 0:
                    values_at_time_2_ema = calculate_ema(values_at_time_2)
                    average_forecasts_2[time_interval] = values_at_time_2_ema.mean()
                else:
                    average_forecasts_2[time_interval] = np.nan

            # APPLY THE FORECAST
            test_data_2['forecast_2'] = test_data_2['time_of_day'].map(average_forecasts_2)

            # CELL 5

            # ACTUAL AND FORECAST VALUES FOR NAIVE 1 and NAIVE 2 
            forecast_naive1 = test_data['forecast'].dropna().values
            forecast_naive2 = test_data_2['forecast_2'].dropna().values

            # ENSURE THAT THE LENGTHS MATCH
            min_len = min(len(forecast_naive1), len(forecast_naive2))
            forecast_naive1 = forecast_naive1[:min_len]
            forecast_naive2 = forecast_naive2[:min_len] 

            # CALCULATE WEIGHTS BASED ON MAPE
            weight_naive1 = 0.58
            weight_naive2 = 0.42  

            print(f"Weight of NAIVE 1: {weight_naive1:.2f}")
            print(f"Weight of NAIVE 2: {weight_naive2:.2f}")


            # INTEGRATION USING WEIGHTED AVERAGE APPROACH
            test_data_2['forecast_3'] = (weight_naive1 * test_data['forecast'].dropna()) + (weight_naive2 * test_data_2['forecast_2'].dropna())

            # CELL 6
            # ENDOGENOUS VARIABLE: INTERSTITIAL GLUCOSE VALUE
            df_cleaned = df_cleaned[~df_cleaned.index.duplicated(keep='first')]
            start = df_cleaned.index.min()
            end = df_cleaned.index.max()
            dates = pd.date_range(start=start, end=end, freq='5min')
            df_cleaned = df_cleaned.reindex(dates)
            df_cleaned['Interstitial Glucose Value'] = df_cleaned['Interstitial Glucose Value'].bfill()

            # TRIM THE DATA FROM THE START UNTIL IT IS DIVISIBLE BY 288
            num_elements = df_cleaned['Interstitial Glucose Value'].shape[0]
            remainder = num_elements % 288
            if remainder > 0:
                df_cleaned = df_cleaned.iloc[remainder:]

            # GROUP DATA BY DAY 
            df_cleaned['Day'] = df_cleaned.index.date
            glucose_by_day = df_cleaned.groupby('Day')['Interstitial Glucose Value'].apply(list)
            glucose_by_day = glucose_by_day.apply(lambda x: x[:288] if len(x) > 288 else x + [np.nan] * (288 - len(x)))
            glucose_array = np.array(glucose_by_day.tolist())

            # Initial dataframe with df_cleaned data
            initial_data = pd.DataFrame({
                'Time': df_cleaned.index,
                'Glucose Level': df_cleaned['Interstitial Glucose Value'],
                'Type': ['Interstitial Glucose'] * len(df_cleaned)
            })

            # Create the initial Altair chart with df_cleaned data
            initial_chart = alt.Chart(initial_data).mark_line().encode(
                x='Time:T',
                y='Glucose Level:Q',
                color=alt.Color('Type:N', scale=alt.Scale(domain=['Interstitial Glucose'], range=['#529acc']))  # Color for chart
            ).properties(
                title='Interstitial Glucose Levels (Initial Data)'
            )

            with chart_section_placeholder:
                next_update_placeholder.empty()  # Clear the placeholder
                with st.spinner('Processing glucose data...'):
                    st.toast("Processing glucose data...", icon="📈")
                    
                    if st.session_state.data_point_counter == 0:
                        # Display the empty chart
                        chart_placeholder.altair_chart(initial_chart, use_container_width=True)

                    # EXOGENOUS VARIABLE: CARBS/SUGAR INTAKE
                    num_nat_dates = current_sugar_df['Date'].isna().sum()
                    if num_nat_dates > 0:
                        print(f"Warning: {num_nat_dates} dates could not be parsed and are NaT. These will be handled.")
                        current_sugar_df['Date'] = current_sugar_df['Date'].ffill()

                    current_sugar_df.set_index('Date', inplace=True)
                    current_sugar_df= current_sugar_df.groupby(current_sugar_df.index).mean()
                    current_sugar_df= current_sugar_df.reindex(df_cleaned.index, method='nearest', fill_value=0)

                    cleaned_glucose_array = remove_outliers_and_replace_with_median(glucose_array)
                    daily_sugar_intake = current_sugar_df.resample('D').mean().fillna(0).values


                    # CELL 7
                    sugar_intake_lagged = current_sugar_df.shift(1)
                    sugar_intake = sugar_decay(sugar_intake_lagged).fillna(0)
                    df.loc[:, 'SugarIntake'] = sugar_intake

                    # CREATE LAGS FOR GLUCOSE
                    for lag in range(1, 13):
                        df[f'Glucose_Lag_{lag}'] = df['Interstitial Glucose Value'].shift(lag)
                    df.dropna(inplace=True)

                    # CREATE TIME-BASED FEATURES
                    df['Hour'] = df.index.hour
                    df['DayOfWeek'] = df.index.dayofweek

                    # ADD CYCLICAL TIME FEATURES
                    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
                    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
                    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
                    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

                    # SPLIT DATA
                    X = df.drop(columns=['Interstitial Glucose Value'])
                    y = df['Interstitial Glucose Value']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                    # DMATRIX FOR XGB
                    dtrain = xgb.DMatrix(X_train, label=y_train)
                    dtest = xgb.DMatrix(X_test, label=y_test)

                    # DEFINE FIXED HYPERPARAMETERS
                    params = {
                        'objective': 'reg:squarederror',
                        'eval_metric': 'rmse',
                        'colsample_bytree': 1.0
                    }

                    # DEFINE VALUES FOR HYPERPARAMETER TUNING
                    param_grid = {
                        'eta': [0.1, 0.15, 0.2],
                        'max_depth': [8, 9, 10],
                        'subsample': [0.8, 1.0],
                    }

                    # INITIALIZE THE XGBREGRESSOR
                    xgb_model = xgb.XGBRegressor(**params, n_estimators=400)

                    # SET UP GRIDSEARCHCV
                    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                                            scoring='neg_mean_squared_error', cv=3, verbose=1)

                    # FIT THE MODEL
                    grid_search.fit(X_train, y_train)

                    print(f"Best Parameters: {grid_search.best_params_}")

                    # GET THE BEST MODEL FROM GRID SEARCH
                    best_xgb_model = grid_search.best_estimator_

                    # RECURSIVE FORECASTING: 288 PAST VALUES
                    n_steps = 288
                    last_row = df.iloc[-1].copy()
                    forecast_values = []

                    # PREDICT GLUCOSE LEVELS IN n STEPS
                    for step in range(n_steps):
                        # ENSURE THAT INPUT_FEATURES ONLY INCLUDES COLUMNS FROM THE TRAINED MODEL
                        input_features = last_row[X_train.columns].values.reshape(1, -1)  

                        # PREDICT GLUCOSE LEVELS USING THE FITTED MODEL 
                        predicted_glucose = float(best_xgb_model.predict(input_features)[0])
                        forecast_values.append(predicted_glucose)

                        # UPDATE GLUCOSE LAGS
                        for lag in range(12, 1, -1):
                            last_row[f'Glucose_Lag_{lag}'] = last_row[f'Glucose_Lag_{lag-1}']
                        
                        last_row['Glucose_Lag_1'] = predicted_glucose 

                        # UPDATE TIME FEATURES 
                        last_row['Hour'] = (last_row['Hour'] - 1) % 24  
                        if last_row['Hour'] == 23:
                            last_row['DayOfWeek'] = (last_row['DayOfWeek'] - 1) % 7

                        # UPDATE CYCLICAL TIME FEATURES
                        last_row['Hour_sin'] = np.sin(2 * np.pi * last_row['Hour'] / 24)
                        last_row['Hour_cos'] = np.cos(2 * np.pi * last_row['Hour'] / 24)
                        last_row['DayOfWeek_sin'] = np.sin(2 * np.pi * last_row['DayOfWeek'] / 7)
                        last_row['DayOfWeek_cos'] = np.cos(2 * np.pi * last_row['DayOfWeek'] / 7)

                    # GENERATE BACKWARD FORECAST FROM THE LAST AVAILABLE TIMESTAMP
                    last_timestamp = df_cleaned.index[-1]  
                    future_dates = pd.date_range(end=last_timestamp, periods=288, freq='5min')

                    # CREATE DATAFRAME FOR PREDICTED VALUES ONLY
                    forecast_df = pd.DataFrame(forecast_values[::-1], index=future_dates, columns=['Predicted Glucose'])

                    # CELL 8

                    # CALCULATE THE RESIDUALS
                    forecast_df_subset = forecast_df.copy()  
                    forecast_df_subset['Residual'] = 0  
                    residuals = forecast_df_subset['Residual'].values.reshape(-1, 1)

                    # SCALE THE RESIDUALS
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_residuals = scaler.fit_transform(residuals)

                    # CREATE SEQUENCES FOR LSTM
                    n = forecast_horizons_list[st.session_state.selected_forecast_horizon_index]["value"]  # FORECAST HORIZONS - USER INPUT

                    X, y = create_dataset(scaled_residuals, time_step)
                    X = X.reshape(X.shape[0], X.shape[1], 1)  

                    # SPLIT THE DATA INTO TRAINING AND TEST SET
                    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

                    # BUILD LSTM MODEL
                    model = Sequential()
                    model.add(Input(shape=(time_step, 1)))  
                    model.add(LSTM(50, return_sequences=True))
                    model.add(LSTM(50))
                    model.add(Dense(1))

                    # COMPILE AND OPTIMIZE MODEL
                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1, clipvalue=1.0), loss='mse')

                    # MODEL TRAINING
                    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                    history = model.fit(X_train, y_train, 
                            epochs=1000, 
                            batch_size=32, 
                            validation_data=(X_val, y_val),  
                            callbacks=[early_stopping])

                    # PREDICT NEXT RESIDUALS
                    predictions = []
                    last_input = scaled_residuals[-time_step:].reshape(1, time_step, 1)

                    for _ in range(n):
                        prediction = model.predict(last_input)
                        
                        # CHECK FOR NaN OR INFINITE VALUES
                        if np.any(np.isnan(prediction)) or np.any(np.isinf(prediction)):
                            raise ValueError("Prediction contains NaN or infinite values.")
                        
                        predictions.append(prediction[0, 0])
                        
                        # UPDATE LAST INPUT
                        last_input = np.append(last_input[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

                    # INVERSE SCALE THE PREDICTIONS
                    predictions = np.array(predictions).reshape(-1, 1)

                    # CLIP PREDICTIONS IF NECESSARY 
                    predictions = np.clip(predictions, -1, 1)  

                    # INVERSE TRANSFORM PREDICTIONS TO ORIGINAL SCALE
                    predictions = scaler.inverse_transform(predictions)

                    # COMPARE THE RESIDUAL FORECAST WITH THE XGBOOST
                    future_forecast = forecast_df['Predicted Glucose'].values[:n] + predictions.flatten()

                    # CREATE DATA FRAME FOR FUTURE PREDICTED VALUES
                    future_index = pd.date_range(start=forecast_df.index[-1] + pd.Timedelta(minutes=5), periods=n, freq='5min')

                    # CREATE THE FUTURE DF WITH CORRECTED INDEX
                    future_df = pd.DataFrame({'Future Predicted Glucose': future_forecast}, index=future_index)

                    # OUTPUT FUTURE PREDICTIONS
                    print(future_df)
                    print(f"Size of future_df: {future_df.size}")

                    # CELL 10

                    # EXTRACT LSTM PREDICTED VALUES FOR THE NEXT N VALUES
                    lstm_predicted_values = future_df['Future Predicted Glucose'].values[:n] 

                    # EXTRACT COMBINED FORECAST FOR THE SAME N VALUES
                    combined_forecast_1 = test_data_2['forecast_3'].values[:n]  

                    # MEAN WEIGHTS
                    weight_lstm = 0.32 
                    weight_combined = 0.68 

                    print(f"lstm weight: {weight_lstm:.2f}%")
                    print(f"naive weight: {weight_combined:.2f}%")

                    # INTEGRATION USING WEIGHTED AVERAGE APPROACH
                    new_combined_forecast = (weight_lstm * lstm_predicted_values) + (weight_combined * combined_forecast_1)

                    # CREATE A NEW DATAFRAME FOR THE NEW COMBINED FORECAST
                    future_index = pd.date_range(start=forecast_df.index[-1], periods=n, freq='5min')
                    new_combined_df = pd.DataFrame({'New Combined Forecast': new_combined_forecast}, index=future_index)

                    # CELL 11
                    # PLOT (USING ALTAIR)
                    # Combine both actual glucose and forecasted glucose into a single dataframe for Altair
                    combined_data = pd.DataFrame({
                        'Time': pd.concat([pd.Series(df_cleaned.index), pd.Series(new_combined_df.index)]).reset_index(drop=True),
                        'Glucose Level': pd.concat([df_cleaned['Interstitial Glucose Value'], new_combined_df['New Combined Forecast']]).reset_index(drop=True),
                        'Type': ['Interstitial Glucose'] * len(df_cleaned) + ['Forecasted'] * len(new_combined_df)
                    })

                    # Create the Altair line chart with custom colors
                    alt_chart = alt.Chart(combined_data).mark_line().encode(
                        x='Time:T',
                        y='Glucose Level:Q',
                        color=alt.Color('Type:N', scale=alt.Scale(domain=['Interstitial Glucose', 'Forecasted'], 
                                                                range=['#529ACC', '#854053']))  # Set specific colors (#1f77b4 - dark blue, #ff7f0e - dark orange)
                    ).properties(
                        title='Interstitial Glucose Levels (With Forecasted Data)'
                    ).interactive()

            st.toast("Done forecasting!", icon="✅")
            # Display the final chart
            chart_section_placeholder.write('<h3>Forecasted Glucose Levels</h3>', unsafe_allow_html=True)
            chart_placeholder.altair_chart(alt_chart, use_container_width=True)

            data_point_counter += 1  # Increment the counter after adding a data point
            st.session_state.data_point_counter = data_point_counter  # Update the session state
            st.session_state.cumulative_added_data = cumulative_added_data  # Update the session state
            
            # Sleep for 5 minutes to simulate interval (change to 5 seconds for testing)
            for remaining in range(interval_minutes * 60, 0, -1):
                next_update_placeholder.info(f"Next update in {remaining // 60}:{remaining % 60:02d}")
                time.sleep(1)  # Sleep for 1 second during countdown


            # Calculate elapsed time and check if we should continue
            elapsed_time = (dt.datetime.now() - start_time).total_seconds() // 60

        # Once all points are added, display the final chart
        next_update_placeholder.success("Simulation complete! Final data processed and displayed.")
        st.session_state.is_running = False  # Stop the simulation


### CONTENT
## !TITLE
st.markdown('<h2><span style="color:#004AAD;">INSIGHT: </span>Integrated Naïve Forecasting Systems for Interstitial Glucose Forecasting—Hybridized with Machine and Deep Learning Techniques</h2>', unsafe_allow_html=True)

## !PLACEHOLDER FOR MESSAGES
message_placeholder = st.empty()

## !FILE UPLOAD SECTION
st.write('<h4>Upload CSVs</h4>', unsafe_allow_html=True)

col_1, col_2 = st.columns(2)
with col_1:
    uploaded_glucose_file = st.file_uploader("Choose the **Glucose Data** CSV file", type="csv")
with col_2:
    uploaded_sugar_file = st.file_uploader("Choose the **Sugar Intake** CSV file", type="csv")

if uploaded_glucose_file and uploaded_sugar_file:
    # Read the CSV files
    try:
        # File upload success message
        message_placeholder.success("Files uploaded successfully!")

        # Load the CSV files (Glucose and Sugar Intake)
        df = pd.read_csv(uploaded_glucose_file, index_col='Datetime', dtype={'Datetime': 'object'})
        df.index = pd.to_datetime(df.index, format='%d/%m/%Y %H:%M', errors='coerce')

        sugar_df = pd.read_csv(uploaded_sugar_file)
        sugar_df['Date'] = pd.to_datetime(sugar_df['Date'], format='mixed', dayfirst=True, errors='coerce') # CONVERT DATA WITH FLEXIBLE PARSING

         # Initialize session state for simulation
        if 'current_data' not in st.session_state:
            st.session_state.current_data = df.iloc[:-12]  # Store initial data excluding the last hour (12 points for 5 mins interval)
            st.session_state.last_hour_data = df.iloc[-12:]  # Last 1 hour of data

        # Initialize sugar_df in session state
        if 'sugar_df' not in st.session_state:
            # Initial sugar_df structure
            st.session_state.sugar_df = sugar_df.copy()


        ## !USER FOOD INTAKE SECTION (USER INPUT)
        # Dropdown for food items and user input for the amount in grams
        with st.form("sugar_input_form"):  # Use a form to prevent immediate reruns
            st.write('<h4>User Food Intake</h4>', unsafe_allow_html=True)

            col_1, col_2 = st.columns(2)
            with col_1:
                st.session_state.selected_food = st.selectbox("Select a food item:", sorted(list(food_dict.keys())), 
                                                    index=sorted(list(food_dict.keys())).index(st.session_state.selected_food))

            with col_2:
                st.session_state.amount_in_grams = st.number_input('Amount (grams):', min_value=0.0, max_value=1000.0, step=0.1, value=st.session_state.amount_in_grams)
            
            submitted = st.form_submit_button("Add Sugar Intake", use_container_width=True)
    
            if submitted:
                add_sugar_intake(food_dict[st.session_state.selected_food] * st.session_state.amount_in_grams / 100)

        ## !FORECAST HORIZONS SECTION (USER INPUT)
        with st.container(border=True):
            # Set the default forecast horizon index
            if 'selected_forecast_horizon_index' not in st.session_state:
                st.session_state.selected_forecast_horizon_index = default_forecast_horizon_index
            
            st.write('<h4>Forecast Horizons</h4>', unsafe_allow_html=True)
            st.write('Select the forecast horizon for the glucose levels:')

            col_1, col_2, col_3, col_4 = st.columns(4)

            with st.container():
                with col_1:
                    five_min_button = st.button(forecast_horizons_list[0]["label"], on_click=set_selected_forecast_horizon_index, args=(0,), use_container_width=True)
                with col_2:
                    fifteen_min_button = st.button(forecast_horizons_list[1]["label"], on_click=set_selected_forecast_horizon_index, args=(1,), use_container_width=True)
                with col_3:
                    thirty_min_button = st.button(forecast_horizons_list[2]["label"], on_click=set_selected_forecast_horizon_index, args=(2,), use_container_width=True)
                with col_4:
                    one_hour_button = st.button(forecast_horizons_list[3]["label"], on_click=set_selected_forecast_horizon_index, args=(3,), use_container_width=True)

            with st.container():
                with col_1:
                    three_hour_button = st.button(forecast_horizons_list[4]["label"], on_click=set_selected_forecast_horizon_index, args=(4,), use_container_width=True)
                with col_2:
                    six_hour_button = st.button(forecast_horizons_list[5]["label"], on_click=set_selected_forecast_horizon_index, args=(5,), use_container_width=True)
                with col_3:
                    twelve_hour_button = st.button(forecast_horizons_list[6]["label"], on_click=set_selected_forecast_horizon_index, args=(6,), use_container_width=True)
                with col_4:
                    twenty_four_hour_button = st.button(forecast_horizons_list[7]["label"], on_click=set_selected_forecast_horizon_index, args=(7,), use_container_width=True)
        
            st.write(f"**Selected forecast horizon:** {forecast_horizons_list[st.session_state.selected_forecast_horizon_index]['label']}")

        ## !PLACEHOLDER FOR THE FORECASTED GLUCOSE LEVELS CHART
        next_update_placeholder = st.empty()  # This will hold the "Next update in..." message
        chart_section_placeholder = st.empty()
        chart_placeholder = st.empty()

        # Run the forecast when the button is clicked
        if st.button("Forecast Glucose Levels", use_container_width=True, type='primary'):
            st.session_state.is_running = True
            # Run the simulation to add new data points every 5 minutes
            simulate_data_addition_with_forecasting(st.session_state.current_data, st.session_state.sugar_df, st.session_state.last_hour_data, interval_minutes=1)


    except Exception as e:
        message_placeholder.error("Please double check that you have uploaded the right CSVs.")
        st.error(f"Error processing files: {e}",)

else:
    message_placeholder.info("Please upload both Glucose Data and Sugar Intake CSV files.")