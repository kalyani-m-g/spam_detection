import streamlit as st
import torch
import joblib

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Movie Genre Classifier",
    page_icon="🎬",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #

MODEL_NAME = "kalyani-m-g/movie-genre-classifier"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

label_encoder = joblib.load("label_encoder.pkl")

# ---------------- HEADER ---------------- #

st.markdown("""
<h1 style='text-align:center;color:#7C3AED;'>
🎬 Movie Genre Classification
</h1>

<p style='text-align:center;font-size:18px;color:gray;'>
AI Powered Movie Genre Prediction using DistilBERT
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- INPUT ---------------- #

st.subheader("📝 Enter Movie Plot")

plot = st.text_area(
    "",
    height=250,
    placeholder="Paste a movie description or plot summary..."
)

# ---------------- PREDICT ---------------- #

if st.button("🎯 Predict Genre"):

    if plot.strip() == "":
        st.warning("Please enter a movie plot.")
    else:

        inputs = tokenizer(
            plot,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        with torch.no_grad():
            outputs = model(**inputs)

        prediction = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

        genre = label_encoder.inverse_transform(
            [prediction]
        )[0]

        confidence = torch.softmax(
            outputs.logits,
            dim=1
        )[0][prediction].item()

        st.markdown("---")

        st.success(
            f"🎭 Predicted Genre: {genre}"
        )

        st.metric(
            "Confidence Score",
            f"{confidence*100:.2f}%"
        )

# ---------------- EXAMPLES ---------------- #

st.markdown("---")

st.subheader("🎞 Example Plots")

st.info("""
A young wizard discovers magical powers and
attends a school of wizardry while battling
dark forces.
""")

st.info("""
Two detectives investigate a mysterious murder
that leads them into a dangerous criminal conspiracy.
""")

st.info("""
A group of astronauts travel through a wormhole
to save humanity from extinction.
""")

# ---------------- FOOTER ---------------- #

st.markdown("---")

st.caption(
    "Movie Genre Classification System | Developed by Kalyani M G"
)
