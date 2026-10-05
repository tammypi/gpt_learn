import pandas as pd
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from gpt import GPTModel
from use_gpt2 import load_weights_into_gpt, GPT_CONFIG_124M
from gpt import generate_text_simple
from gpt_download import download_and_load_gpt2, text_to_token_ids, token_ids_to_text

tokenizer = tiktoken.get_encoding("gpt2")

class SampDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

train_dataset = SampDataset("./train.csv", max_length=None, tokenizer=tokenizer)
validation_dataset = SampDataset("./validation.csv", max_length=None, tokenizer=tokenizer)
test_dataset = SampDataset("./test.csv", max_length=None, tokenizer=tokenizer)

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True, 
    num_workers=num_workers,
    drop_last=True
)
validation_loader = DataLoader(
    dataset=validation_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False
)

settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)

gpt = GPTModel(GPT_CONFIG_124M)
load_weights_into_gpt(gpt, params)
gpt.to("cuda")

# 将线性输出层替换，之前是从embed_dim映射到词表大小，现在则改为映射到2 

torch.manual_seed(123)
num_classes = 2 
gpt.out_head = torch.nn.Linear(
    in_features=GPT_CONFIG_124M["emb_dim"],
    out_features=num_classes
)

# 让最终归一化层和最后一个transformers块可训练
for param in gpt.trf_blocks[-1].parameters():
    param.requires_grad = True 
for param in gpt.final_norm.parameters():
    param.requires_grad = True 

inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
print(inputs.shape, inputs)

with torch.no_grad():
    outputs = gpt(inputs)

print(outputs.shape, outputs)