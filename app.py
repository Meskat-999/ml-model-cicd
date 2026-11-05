
import gradio as gr
from transformers import pipeline

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

def analyze_sentiment(text):
    if not text.strip():
        return "Please enter some text."

    result = sentiment_model(text)[0]

    label = result["label"]
    score = round(result["score"] * 100, 2)

    return f"Sentiment: {label}\nConfidence: {score}%"

app = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=3, label="Enter your text"),
    outputs=gr.Textbox(label="Sentiment Result"),
    title="Sentiment Analysis System",
    description="Analyze text using a pre-trained DistilBERT model."
)

app.launch()
