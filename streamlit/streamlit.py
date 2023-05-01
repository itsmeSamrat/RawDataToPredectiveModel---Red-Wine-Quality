import pickle5 as pickle
import pandas as pd
import streamlit as st


import warnings

warnings.simplefilter("ignore")


def predict_wine_quality(
    fixed_acidity,
    citric_acid,
    sulphates,
    alcohol,
    volatile_acidity,
    chlorides,
    density,
    total_sulfur_dioxide,
):
    user_data = {
        "fixed acidity": fixed_acidity,
        "citric acid": citric_acid,
        "sulphates": sulphates,
        "alcohol": alcohol,
        "volatile acidity": volatile_acidity,
        "chlorides": chlorides,
        "density": density,
        "total sulfur dioxide": total_sulfur_dioxide,
    }
    data = pd.DataFrame(user_data, index=[0])

    # loading the model from

    rf_model = pickle.load(open("final_trained_model.pkl", "rb"))

    # loading the scaler values

    rf_scaler = pickle.load(open("final_trained_scaler_model.pkl", "rb"))

    data_scaled = rf_scaler.transform(data)

    predict = rf_model.predict(data_scaled)
    predict = predict[0]
    return data, predict


# def main():
st.title("Red Wine Quality Prediction")
html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style = "color:white;text-align:center;">Red Wine Quality Prediction</h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)
fixed_acidity = st.number_input("Fixed Acidity")
citric_acid = st.number_input("Citric Acid")
sulphates = st.number_input("Sulphates")
alcohol = st.number_input("Alcohol")
volatile_acidity = st.number_input("Volatile Acidity")
chlorides = st.number_input("Chlorides")
density = st.number_input("Density")
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide")
result = 0

if st.button("Predict"):
    result = predict_wine_quality(
        fixed_acidity,
        citric_acid,
        sulphates,
        alcohol,
        volatile_acidity,
        chlorides,
        density,
        total_sulfur_dioxide,
    )[1]
st.write(
    predict_wine_quality(
        fixed_acidity,
        citric_acid,
        sulphates,
        alcohol,
        volatile_acidity,
        chlorides,
        density,
        total_sulfur_dioxide,
    )[0]
)

st.success(
    f"The Wine Quality with the data provided is predicted to be {round(result)}"
)


# if __name__ == "__main__":
#     main()
