import streamlit as st
import pickle
import gzip

# Set the page configuration for Streamlit
st.set_page_config(page_title="Timelytics", layout="wide")

# Function to load the compressed model from a .gz file
@st.cache_resource  # Use caching to avoid reloading the model every time
def load_model():
    with gzip.open('voting_model.pkl.gz', 'rb') as f:
        model = pickle.load(f)
    return model

# Load the model
voting_model = load_model()

# Sample UI to test predictions (this part depends on your specific use case)
st.title("Predict Wait Time with Voting Regressor Model")

# Add user inputs for model features (modify based on your actual input features)
purchase_dow = st.number_input("Purchase Day of the Week", min_value=0, max_value=6, value=3)
purchase_month = st.number_input("Purchase Month", min_value=1, max_value=12, value=1)
year = st.number_input("Year", value=2021)
product_size_cm3 = st.number_input("Product Size (cm³)", value=5000)
product_weight_g = st.number_input("Product Weight (g)", value=1500)
geolocation_state_customer = st.number_input("Customer Geolocation State", value=10)
geolocation_state_seller = st.number_input("Seller Geolocation State", value=20)
distance = st.number_input("Distance (km)", value=300.0)

# Button to trigger the prediction
if st.button("Predict Wait Time"):
    input_data = [[purchase_dow, purchase_month, year, product_size_cm3, product_weight_g, geolocation_state_customer, geolocation_state_seller, distance]]
    
    # Make a prediction using the voting model
    prediction = voting_model.predict(input_data)
    
    # Show the prediction
    st.write(f"Predicted Wait Time: {prediction[0]} days")
