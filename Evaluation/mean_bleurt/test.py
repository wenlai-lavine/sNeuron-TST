from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-large-512")
model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-large-512")
model.eval()

references = ["hello world", "hello world"]
candidates = ["hi universe", "bye world"]

with torch.no_grad():
  scores = model(**tokenizer(references, candidates, return_tensors='pt'))[0].squeeze()
print(scores.tolist()) # tensor([0.9877, 0.0475])