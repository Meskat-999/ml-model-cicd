
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sentiment Analysis")

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )

st.title("Sentiment Analysis System_1")
st.write("Using a Pre-trained DistilBERT Model - CI/CD Test")

text = st.text_area("Enter your text:")

if st.button("Analyze Sentiment"):
    if text.strip():
        with st.spinner("Analyzing..."):
            model = load_model()
            result = model(text)[0]

        st.success("Analysis Completed!")
        st.write("Sentiment:", result["label"])
        st.write("Confidence:", f'{result["score"] * 100:.2f}%')
    else:
        st.warning("Please enter some text.")
