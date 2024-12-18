import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Streamlit app title
st.title("Linear Regression App")

# File uploader to load the CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Show the dataset
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Select X and Y features from the dataset
    columns = df.columns.tolist()

    # Dropdown menus to choose X and Y features
    x_feature = st.selectbox("Select the feature for X", columns)
    y_feature = st.selectbox("Select the feature for Y", columns)

    # Button to trigger the model training and evaluation
    if st.button("Run Linear Regression"):
        if x_feature != y_feature:
            # Prepare the data for modeling
            X = df[[x_feature]]
            y = df[y_feature]

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create a linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Model evaluation metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Display results
            st.write(f"Mean Squared Error (MSE): {mse:.2f}")
            st.write(f"R-squared (R²): {r2:.2f}")

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.scatter(X_test, y_test, color="blue", label="Actual")
            plt.plot(X_test, y_pred, color="red", label="Predicted")
            plt.xlabel(x_feature)
            plt.ylabel(y_feature)
            plt.title(f"Linear Regression: {x_feature} vs {y_feature}")
            plt.legend()
            st.pyplot(plt)

        else:
            st.error("X and Y features cannot be the same!")

           streamlit run linear_regression_app.py
