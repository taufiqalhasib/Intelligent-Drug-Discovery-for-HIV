import streamlit as st

import pickle
import csv
import joblib
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import tensorflow as tf

#CC(=O)N1c2ccccc2Sc2c1ccc1ccccc21

# Load model
#model = joblib.load("ml_model4_SVM.pkl")
#model = pickle.load(open('ml_model4_SVM.pkl', 'rb'))
model = pickle.load(open('DL_model3.pkl', 'rb'))

st.title("Intelligent Drug Discovery for HIV")

# Input for SMILES
smiles_input = st.text_input("Enter a SMILES string:")


# Finger Print Generating
def smiles_to_features(smiles):
    mol1 = Chem.MolFromSmiles(smiles)
    if mol1 is None:
        return None

    # Compute Morgan fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
    arr = np.zeros((1,))
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


if st.button("Predict"):
    features = smiles_to_features(smiles_input)

    # Prediction
    if features is not None:
        features = features.reshape(1, -1)  # 2D array for sklearn
        prediction = model.predict(features)
        st.write("Prediction:", "Active" if prediction == 1 else "Inactive")
        st.write("Predicted Activity:", prediction[0])
    else:
        st.error("Invalid SMILES string")




