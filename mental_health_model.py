import torch
import random

import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class MentalHealthLLM:
    def __init__(self, checkpoint, token):
        # TODO: launch app with LLM loaded with bnb config
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, token=token)
        # self.model = AutoModelForCausalLM.from_pretrained(checkpoint, token=token)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            #quantization_config=bnb_config,
            load_in_4bit=True,
            device_map="auto",
            token=token
        )

    def generate(self, text, device, max_len=200):
        messages = [{
                "role": "user",
                "content": text
        }]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = self.model.generate(**inputs, max_length=max_len, num_return_sequences=1)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text.split("assistant")[1]