import pandas as pd
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from gpt import GPTModel
from use_gpt2 import load_weights_into_gpt, GPT_CONFIG_124M, text_to_token_ids, token_ids_to_text
from gpt import generate_text_simple
from gpt_download import download_and_load_gpt2

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

device = "cuda"

gpt = GPTModel(GPT_CONFIG_124M)
load_weights_into_gpt(gpt, params)

# 将线性输出层替换，之前是从embed_dim映射到词表大小，现在则改为映射到2 

torch.manual_seed(123)
num_classes = 2 
gpt.out_head = torch.nn.Linear(
    in_features=GPT_CONFIG_124M["emb_dim"],
    out_features=num_classes
)
gpt.to(device)

# 让最终归一化层和最后一个transformers块可训练
for param in gpt.trf_blocks[-1].parameters():
    param.requires_grad = True 
for param in gpt.final_norm.parameters():
    param.requires_grad = True 

# inputs = tokenizer.encode("Do you have time")
# inputs = torch.tensor(inputs).unsqueeze(0).to(device)
# print(inputs.shape, inputs)

# with torch.no_grad():
#     outputs = gpt(inputs)

# print(outputs.shape, outputs)

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_preidctions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            with torch.no_grad():
                logits = gpt(input_batch)[:, -1, :]
            predict_labels = torch.argmax(logits, dim=-1)
            num_examples += predict_labels.shape[0]
            correct_preidctions += (
                (predict_labels == target_batch).sum().item()
            )
        else:
            break 
    return correct_preidctions / num_examples

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss 

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss = loss.item()
        else:
            break 
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def train_classifier_simple(model, train_loader, val_loader, optimizer, device,
                            num_epochs, eval_freqs, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device 
            )
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freqs == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} Step {global_step}: "
                        f"Train loss {train_loss:.3f} "
                        f"Val loss {val_loss: .3f} ")
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )
        print(f"train_accuracy: {train_accuracy}")
        print(f"val_accuracy: {val_accuracy}")
        train_accs.append(train_accuracy)
        val_accs.append(val_accs)
    return train_losses, val_losses, train_accs, val_accs, examples_seen

import time 
start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-5, weight_decay=0.1)
num_epochs = 5

train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    gpt, train_loader, validation_loader, optimizer, device,
    num_epochs=num_epochs, eval_freqs=50,
    eval_iter=5 
)

end_time = time.time()
execution_time = (end_time - start_time) / 60
print(f"time waste: {execution_time}")

# 保存模型
torch.save(gpt.state_dict(), "spam_classifier.pth")
print("model saved")