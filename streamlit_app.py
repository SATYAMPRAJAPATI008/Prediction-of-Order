# Import required libraries
import streamlit as st
import pickle
import numpy as np
from PIL import Image
import pandas as pd

# Set the page configuration
st.set_page_config(page_title="Timelytics", page_icon=":pencil:", layout="wide")

# Display title and app description
st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques.")

st.caption(
    """
    Timelytics is an ensemble model that utilizes three powerful machine learning algorithms - XGBoost, 
    Random Forests, and Support Vector Machines (SVM) - to accurately forecast Order to Delivery (OTD) times. 
    By combining the strengths of these algorithms, Timelytics provides robust predictions to help optimize 
    supply chain operations.
    """
)

# Caching the model for faster loading
@st.cache_resource
def load_model():
    # Load the trained ensemble model from a pickle file
    with open("./voting_model.pkl", "rb") as file:
        return pickle.load(file)

# Load the model
voting_model = load_model()

# Prediction function
def wait_time_predictor(purchase_dow, purchase_month, year, product_size_cm3, product_weight_g, 
                        geolocation_state_customer, geolocation_state_seller, distance):
    try:
        # Make a prediction using the loaded model
        prediction = voting_model.predict(np.array(
            [[purchase_dow, purchase_month, year, product_size_cm3, product_weight_g,
              geolocation_state_customer, geolocation_state_seller, distance]]
        ))
        return round(prediction[0])
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None

# Sidebar for input parameters
with st.sidebar:
    img = Image.open("./assets/supply_chain_optimisation.jpg")
    st.image(img)
    st.header("Input Parameters")
    
    purchase_dow = st.number_input("Purchased Day of the Week", min_value=0, max_value=6, step=1, value=3)
    purchase_month = st.number_input("Purchased Month", min_value=1, max_value=12, step=1, value=1)
    year = st.number_input("Purchased Year", value=2018)
    product_size_cm3 = st.number_input("Product Size in cm³", value=9328)
    product_weight_g = st.number_input("Product Weight in grams", value=1800)
    geolocation_state_customer = st.number_input("Geolocation State of the Customer", value=10)
    geolocation_state_seller = st.number_input("Geolocation State of the Seller", value=20)
    distance = st.number_input("Distance", value=475.35)
    submit = st.button(label="Predict Wait Time!")

# Output the prediction result
if submit:
    with st.spinner("Calculating..."):
        prediction = wait_time_predictor(purchase_dow, purchase_month, year, product_size_cm3,
                                         product_weight_g, geolocation_state_customer, 
                                         geolocation_state_seller, distance)
        if prediction is not None:
            st.success(f"Predicted Wait Time: {prediction} days")

# Display a sample dataset
sample_data = {
    "Purchased Day of the Week": [0, 3, 1],
    "Purchased Month": [6, 3, 1],
    "Purchased Year": [2018, 2017, 2018],
    "Product Size in cm³": [37206.0, 63714, 54816],
    "Product Weight in grams": [16250.0, 7249, 9600],
    "Geolocation State Customer": [25, 25, 25],
    "Geolocation State Seller": [20, 7, 20],
    "Distance": [247.94, 250.35, 4.915]
}

df = pd.DataFrame(sample_data)
st.header("Sample Dataset")
st.write(df)
