import os
import torch
import streamlit as st
import matplotlib.pyplot as plt

from sentiment_model import SentimentModel
from mental_health_model import MentalHealthLLM

mental_health_checkpoint = "yaroslava/llama_mental_health_chat"
sentiment_checkpoint = "yaroslava/sentiment-roberta-go-emotions"
HF_TOKEN = os.getenv("HF_TOKEN")

def main():
    sentiment_model = SentimentModel(sentiment_checkpoint, token=HF_TOKEN)
    # mental_health_llm = MentalHealthLLM(mental_health_checkpoint, token=HF_TOKEN)

    st.title("Self-reflection bot")
    st.subheader(
        "Welcome to self-reflection bot, the tool that is aimed to help with self-reflections, to have better understanding of the emotions and feelings that one might feel in certain situations.")
    st.markdown(
        "This tool is intended for initial support and guidance, not a replacement for professional mental health care")

    # get input text from user
    user_text = st.text_area("What's on your mind?")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # make sentiment predictions
    if st.button("Get support"):
        emotion_prediction_df = sentiment_model.predict(user_text, device)
        st.markdown("Emotions you might experience in this situation:")
        fig, ax = plt.subplots()
        emotion_prediction_df.sort_values('prob').plot.barh(ax=ax)
        st.pyplot(fig)

        # mental_health_chat_response = mental_health_llm.generate(user_text, device)
        # st.markdown(mental_health_chat_response)

if __name__ == "__main__":
    main()