import json 
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from functools import partial
from gpt_download import download_and_load_gpt2
from gpt import GPTModel, generate_text_simple
from use_gpt2 import load_weights_into_gpt
from train import text_to_token_ids, token_ids_to_text, calc_loss_loader, train_model_simple

with open("./instruction-data.json", "r") as f:
    data = json.loads(f.read())

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describe a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )

    return instruction_text + input_text


train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion:train_portion+test_portion]
val_data = data[train_portion+test_portion:]

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data 
        self.encoded_texts = []
        self.instruction_lengths = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )
            self.instruction_lengths.append(len(tokenizer.encode(instruction_plus_input)))

    def __getitem__(self, index):
        return self.encoded_texts[index], self.instruction_lengths[index]

    def __len__(self):
        return len(self.data)

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="CUDA"
):
    batch_max_length = max(len(item[0]) for item in batch)
    inputs_lst, targets_lst = [], []

    for (item, instruction_length)  in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # 只让 response 参与 loss（targets 相对 inputs 右移一位，所以减 1）
        if instruction_length is not None:
            targets[:instruction_length - 1] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targests_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targests_tensor

device = "cuda"

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

num_workers = 0
batch_size = 8
tokenizer = tiktoken.get_encoding("gpt2")

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

test_datset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_datset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

BASIC_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "emb_dim": 1024,
    "n_layers": 24,
    "n_heads": 16
}

settings, params = download_and_load_gpt2(
    model_size="355M",
    models_dir="gpt2"
)

model = GPTModel(BASIC_CONFIG)
load_weights_into_gpt(model, params)
model.to(device)
model.eval()

# 训练前打印输出

torch.manual_seed(123)
input_text = format_input(val_data[0])
print("input:\n", input_text)

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASIC_CONFIG["context_length"]
)
print("output:\n", token_ids_to_text(token_ids, tokenizer))

# 训练

import time 
start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.00005, weight_decay=0.1
)
num_epochs = 4

train_losses, val_losses, token_seen = train_model_simple(
    model,
    train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freqs=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

end_time = time.time()
execution_time = (end_time-start_time)/60
print(f"waste time: {execution_time}")

# 训练后用测试集数据测试一下
model.eval()
torch.manual_seed(123)
input_text = format_input(test_data[0])
print("input:\n", input_text)

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASIC_CONFIG["context_length"]
)
print("output:\n", token_ids_to_text(token_ids, tokenizer))