import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="MLP Digits", page_icon="🔢")

@st.cache_resource
def load_model():
    return joblib.load("mlp_digits_pipeline.joblib")

model = load_model()

st.title("Reconnaissance de chiffres manuscrits")
st.write("Dessinez un chiffre sur la grille 8 × 8.")

pixels = []

for i in range(8):
    cols = st.columns(8)
    row = []
    for j in range(8):
        value = cols[j].selectbox(
            f"Pixel {i},{j}",
            options=[0, 4, 8, 12, 16],
            index=0,
            key=f"pixel_{i}_{j}",
            label_visibility="collapsed"
        )
        row.append(value)
    pixels.extend(row)

if st.button("Prédire le chiffre"):
    image = np.array(pixels, dtype=float).reshape(1, 64)

    prediction = model.predict(image)[0]
    probabilities = model.predict_proba(image)[0]

    st.success(f"Chiffre prédit : {prediction}")

    fig, ax = plt.subplots()
    ax.imshow(image.reshape(8, 8), cmap="gray", vmin=0, vmax=16)
    ax.axis("off")
    st.pyplot(fig)

    st.subheader("Probabilités estimées")
    st.bar_chart({
        str(i): float(probabilities[i])
        for i in range(10)
    })
