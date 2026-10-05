import pandas as pd
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from gpt import GPTModel
from use_gpt2 import load_weights_into_gpt, GPT_CONFIG_124M, text_to_token_ids, token_ids_to_text
from gpt import generate_text_simple

tokenizer = tiktoken.get_encoding("gpt2")
device = "cuda"

gpt = GPTModel(GPT_CONFIG_124M)
model_state_dict = torch.load("./spam_classifier.pth", map_location="cuda")
gpt.load_state_dict(model_state_dict)

def classify_review(
    text, model, tokenizer, device, max_length=None, pad_token_ids=50256
):
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    input_ids = input_ids[:min(max_length, supported_context_length)]
    input_ids += [pad_token_ids]*(max_length-len(input_ids))
    input_tensor = torch.tensor(
        input_ids, device=device
    ).unsqueeze(0)
    with torch.no_grad():
        logits = gpt(input_tensor)[:,-1,:]
    predict_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predict_label == 1 else "not spam"

text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

print(classify_review(text_1))

