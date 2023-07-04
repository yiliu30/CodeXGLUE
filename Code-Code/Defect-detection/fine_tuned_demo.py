from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

model_path_name="/home/st_liu/workspace/inc_examples/mrm8488/codebert-base-finetuned-detect-insecure-code"

tokenizer = AutoTokenizer.from_pretrained(model_path_name)
model = AutoModelForSequenceClassification.from_pretrained(model_path_name)


import pdb; pdb.set_trace()
inputs = tokenizer("your code here", return_tensors="pt", truncation=True, padding='max_length')
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits

print(np.argmax(logits.detach().numpy()))