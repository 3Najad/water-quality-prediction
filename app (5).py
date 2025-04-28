
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.title('💧 Water Quality Prediction App')

# Load data
data = pd.read_csv('water_potability.csv')
X = data.drop('Potability', axis=1)
y = data['Potability']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

st.header('Enter Water Chemical Properties:')

ph = st.number_input('pH Value', min_value=0.0, max_value=14.0, step=0.1)
Hardness = st.number_input('Hardness', min_value=0.0)
Solids = st.number_input('Solids', min_value=0.0)
Chloramines = st.number_input('Chloramines', min_value=0.0)
Sulfate = st.number_input('Sulfate', min_value=0.0)
Conductivity = st.number_input('Conductivity', min_value=0.0)
Organic_carbon = st.number_input('Organic Carbon', min_value=0.0)
Trihalomethanes = st.number_input('Trihalomethanes', min_value=0.0)
Turbidity = st.number_input('Turbidity', min_value=0.0)

if st.button('Predict Water Quality'):
    input_data = np.array([[ph, Hardness, Solids, Chloramines, Sulfate,
                            Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success('✅ The Water is Safe for Drinking (Potable)')
    else:
        st.error('❌ The Water is Not Safe for Drinking (Not Potable)')
