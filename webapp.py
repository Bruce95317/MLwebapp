# This programme deermien whether a pateint have diabetes using machine learning and python

# import library
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

# Create a title and a ub-title
st.write("""
# Diabetes Detection
Detect if someone has diabetes
""")

# Open and display an image
image = Image.open(
    'adobestock_276205639.png')
st.image(image, caption='ML', use_column_width=True)

# Get the dat
df = pd.read_csv('diabetes.csv')
# Set a subheader
st.subheader('Data Information')
# show the data as a table
st.dataframe(df)
# Show statistics on the data
st.write(df.describe())
# Show the data as a chart
chart = st.bar_chart(df)

# Split the data into independent 'X' an dependent 'Y' as variables
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

# Split the data into 75% Training and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)

# Get the feature input from the user


def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies_months', 0, 17, 3)
    glucose = st.sidebar.slider('glucose_mg/dL', 0, 200, 177)
    blood_pressure = st.sidebar.slider('blood_pressure_mmHg', 0, 150, 57)
    skin_thickness = st.sidebar.slider('skinthickness_0.1mm', 0, 99, 5)
    Insulin = st.sidebar.slider('Insulin', 0.0, 900.0, 0.0)
    BMI = st.sidebar.slider('BMI', 0.0, 70.0, 20.3)
    DPF = st.sidebar.slider(
        'DiabetesPedigreeFunction_DPF', 0.078, 2.42, 0.3753)
    age = st.sidebar.slider('age', 0, 99, 30)


# Store a dictionary into a variable
    user_data = {'pregnancies_months': pregnancies,
                 'glucose_mg/dL': glucose,
                 'blood_pressure_mmHg': blood_pressure,
                 'skinthickness_0.1mm': skin_thickness,
                 'Insulin': Insulin,
                 'BMI': BMI,
                 'DiabetesPedigreeFunction_DPF': DPF,
                 'age': age
                 }

# Transform the data into a dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features


# Transform the data into a dataframe
user_input = get_user_input()

# Set a subheader and display the users input
st.subheader('User Input: ')
st.write(user_input)

# Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

# Show the model metrics
st.subheader('Model Test Accurancy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%')
# Store the models prediction in a variable
prediction = RandomForestClassifier.predict(user_input)

# Set a subheader to dispaly the classifiction
st.subheader('Classification: ')
st.write(prediction)
