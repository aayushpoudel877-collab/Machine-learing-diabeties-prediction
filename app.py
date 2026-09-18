import streamlit as st
from PIL import Image
from sentinel_vision.predict import Predictor
st.set_page_config(page_title="SentinelVision",page_icon="♻️")
st.title("SentinelVision ♻️"); st.caption("Waste image classification — educational prototype")
file=st.file_uploader("Upload a waste image",type=["jpg","jpeg","png","webp"])
if file:
    image=Image.open(file).convert("RGB"); st.image(image,caption="Input",use_container_width=True)
    try:
        result=Predictor().predict(image,top_k=6); st.metric("Prediction",result["prediction"],f"{result['confidence']*100:.1f}% confidence"); st.subheader("Class probabilities")
        for item in result["top_k"]: st.write(f"**{item['class']}** — {item['probability']*100:.1f}%"); st.progress(item["probability"])
    except FileNotFoundError: st.warning("No trained model found. Train it first using README.md.")
