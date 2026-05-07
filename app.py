import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Granja Armengol — Brand Sentiment", page_icon="🐄")
st.title("🐄 Granja Armengol — Sentiment Analyzer")
st.markdown("Analyzes Instagram captions in Catalan, Spanish or English.")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment")

def convert_stars(label):
    stars = int(label.split()[0])
    if stars >= 4:
        return "POSITIVE", "green"
    elif stars == 3:
        return "NEUTRAL", "gray"
    else:
        return "NEGATIVE", "red"

sent = load_model()
text = st.text_area("Paste an Instagram caption:", height=150)
if text:
    res = sent(text)[0]
    label, color = convert_stars(res["label"])
    st.markdown(f"### :{color}[{label}] · confidence {res['score']:.2f}")
