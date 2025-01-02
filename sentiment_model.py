import torch
import random

import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from datasets import load_dataset

class SentimentModel:
    def __init__(self, checkpoint, token):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, token=token)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, token=token)
        go_emotions_ds = load_dataset("google-research-datasets/go_emotions", "simplified")
        self.label_names = go_emotions_ds['train'].features['labels'].feature.names

    def predict(self, text, device, top_n=5):
        tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
        output = self.model(**tokens)
        with torch.no_grad():
            probs = torch.softmax(output.logits, dim=1)

        d = {self.label_names[i]: prob * 100 for i, prob in enumerate(probs.tolist()[0])}
        df = pd.DataFrame(d.items(), columns=['emotion', 'prob'])
        predictions_df = df.sort_values('prob', ascending=False).head(top_n).set_index('emotion')
        return predictions_df