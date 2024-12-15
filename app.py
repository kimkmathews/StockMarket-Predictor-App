import streamlit as st
import pandas as pd
from data_processing import process_split_data, train_evaluate_classification_model,train_evaluate_regression_model, generate_features
import matplotlib.pyplot as plt

# Set page title and icon
#st.set_page_config(page_title="Stock Price Prediction App", page_icon="📈")
st.set_page_config(
    page_title="Stock Price Prediction App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Title and Introduction
st.title("Stock Price Prediction App")
st.write("This app allows you to predict the direction of a stock price based on historical data.")

# User Input Sidebar
with st.sidebar:
    # Stock Symbol Input
    stock_symbol = st.text_input("Enter Stock Symbol (NSE)", "RELIANCE")

    # Target Variable Options (Adapt based on classification or regression)
    #target_variable_type = st.radio("Target Variable Type", ("Classification", "Regression"))
    target_variable_type = "Regression"
    if target_variable_type == "Classification":
        target_days_options = st.selectbox("Target Days", [1, 3, 5, 7, 10])
        target_percent_options = st.selectbox("Target Percent Change", [5, 7.5, 10, 12.5, 15])
        target_percent_options = target_percent_options / 100
        model_options = st.selectbox("Select Model", [
            "LogisticRegression", "RandomForestClassifier", "SVC", "GradientBoostingClassifier", "XGBoostClassifier"
        ])
    else:
        target_days_options = st.selectbox("Target Days", [1, 3, 5, 7, 10])
        model_options = st.selectbox("Select Model", [
            "LinearRegression", "RandomForestRegressor", "SVR", "GradientBoostingRegressor", "XGBRegressor"
        ])
    
        
    # Run Button
    run_button = st.button("Run Prediction")
if not run_button:
    st.write("**Feature Engineering:**")
    st.write("- **Moving Averages:** Calculated using different window sizes (e.g., 7, 14, 21 days) to identify trends.")
    st.write("- **Technical Indicators:** Incorporated RSI, MACD, Bollinger Bands, Stochastic, and Williams R to capture market momentum and volatility.")
    st.write("- **Ratios with Close:** Calculated ratios between close price and other features like open, high, low, and volume to identify price patterns.")
    st.write("- **Time Series Features:** Created lagged features for the past 21 days to capture historical patterns.")


# Functionality based on user input
if run_button:
    # Process and Split Data
    try:
        if target_variable_type == "Classification":
            scaled_X_train, y_train, scaled_X_test, y_test, scaled_X_val, merged_df, scaler = process_split_data(
                stock_symbol, target_days_options, target_percent_options,target_variable_type
            )
        else:
            scaled_X_train, y_train, scaled_X_test, y_test, scaled_X_val, merged_df, scaler = process_split_data(
                stock_symbol, target_days_options,0,target_variable_type
            )
    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.stop()
# Train and Evaluate Model
    if target_variable_type == "Classification":
        model, symbol, accuracy, precision, recall, roc_auc, f1 = train_evaluate_classification_model(
            stock_symbol, scaled_X_train, y_train, scaled_X_test, y_test, scaler, model_engine=model_options
        )
        validations = model.predict(scaled_X_val)
        merged_df['Predictions'] = validations
        # Display classification results
        results_data = {
            "Stock Symbol": symbol,
            "Target Days": target_days_options,
            "Target Percent Change": f"{target_percent_options}%",
            "Selected Model": model_options,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "ROC AUC": roc_auc,
            "F1 Score": f1
        }
        st.subheader("Model Performance")
        results_df = pd.DataFrame(results_data, index=[0])
        st.table(results_df)
        st.subheader("Model Result")
        merged_df = merged_df.round(2)
        st.table(merged_df)
    else:
        model, symbol, rmse, r2, mae, mape = train_evaluate_regression_model(
            stock_symbol, scaled_X_train, y_train, scaled_X_test, y_test, scaler, model_engine=model_options
        )
        predictions_test = model.predict(scaled_X_test)
        predictions = model.predict(scaled_X_val)
        merged_df['Predictions'] = predictions
        # Display regression results
        results_data = {
            "Stock Symbol": symbol,
            "Target Days": target_days_options,
            "Selected Model": model_options,
            "RMSE": rmse,
            "R-squared": r2,
            "Mean Absolute Error":mae,
            "Mean Absolute Percentage Error":mape}
        st.subheader("Model Performance")
        results_df = pd.DataFrame(results_data, index=[0])
        st.table(results_df)
        st.subheader("Model Result")
        merged_df = merged_df.round(2)
        st.table(merged_df)
        # Plot y_test, y_test_pred, y_val, and y_val_pred
        test_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions_test})
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot your data
        ax.plot(test_df.index, test_df['Actual'], label='Test Close')
        ax.plot(test_df.index, test_df['Predicted'], label='Test Predictions')
        ax.plot(merged_df.index, merged_df['Target'], label='Validation Close')
        ax.plot(merged_df.index, merged_df['Predictions'], label='Validation Predictions')
        ax.legend()
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.set_title('Actual vs Predicted Values')

        # Display the plot in Streamlit
        st.pyplot(fig)