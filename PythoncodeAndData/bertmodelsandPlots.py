from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report
import torch
import matplotlib.pyplot as plt
import numpy as np
import logging
from itertools import product

def loadMyData(myFile):
    with open(myFile, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    X = [lines[i] for i in range(0, len(lines)-1, 2)]
    y = [int(lines[i+1]) for i in range(0, len(lines)-1, 2)]
    return X, y

X_train, y_train = loadMyData("train.txt")
X_test, y_test = loadMyData("test.txt")

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        return {**encoding, "labels": torch.tensor(label)}

BertModels = [
    "prajjwal1/bert-tiny",
    "prajjwal1/bert-mini",
    "Davlan/afro-xlmr-small"
]

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Batchez = 8
Epocs = 3
RateLearn = 2e-5
loraVal = [4]
LoraVal2 = [8]
DropVals = [0.1]
Targets = [["query", "value"]]
HyperParams = list(product(loraVal, LoraVal2, DropVals, Targets))

def plot_attention_map(attentions, model_name, lora=False, tokenizer=None, sample_text=None):
    attn_layers = torch.stack(attentions)
    attn_avg = attn_layers[:, 0].mean(dim=1).mean(dim=0)
    attn_np = attn_avg.cpu().numpy()
    tokens = tokenizer.tokenize(sample_text)
    tokens = tokens[:min(len(tokens), attn_np.shape[0])]

    attn_np = attn_np[:len(tokens), :len(tokens)]

    plt.figure(figsize=(12, 10))
    plt.imshow(attn_np, cmap='viridis', interpolation='nearest', aspect='equal')
    plt.colorbar()
    plt.xticks(ticks=np.arange(len(tokens)), labels=tokens, rotation=90, fontsize=8)
    plt.yticks(ticks=np.arange(len(tokens)), labels=tokens, fontsize=8)

    title = f"Attention Map - {model_name} {'(LoRA)' if lora else '(No LoRA)'}"
    plt.title(title)
    plt.xlabel("Token Position")
    plt.ylabel("Token Position")
    plt.tight_layout()

    filename = f"attention_{model_name.replace('/', '_')}_{'lora' if lora else 'nolora'}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved zoomed-in attention heatmap to {filename}")

for bertModelInstance in BertModels:
    for Lora in [False]:
        mode = "with LoRA" if Lora else "without LoRA"
        print(f"\nTraining {bertModelInstance} {mode}...\n")

        tokenizer = AutoTokenizer.from_pretrained(bertModelInstance)
        model = AutoModelForSequenceClassification.from_pretrained(
            bertModelInstance,
            num_labels=2,
            output_attentions=True
        )
        model.to(device)

        if Lora:
            for r, alpha, Dropout, targets in HyperParams:
                ConfigLora = LoraConfig(
                    r=r,
                    lora_alpha=alpha,
                    lora_dropout=Dropout,
                    target_modules=targets,
                    bias="none",
                )
                model = get_peft_model(model, ConfigLora)
                break

        TrainData = TextDataset(X_train, y_train, tokenizer)
        TestData = TextDataset(X_test, y_test, tokenizer)
        train_loader = DataLoader(TrainData, batch_size=Batchez, shuffle=True)
        test_loader = DataLoader(TestData, batch_size=Batchez)

        optimizer = torch.optim.AdamW(model.parameters(), lr=RateLearn)
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*Epocs)

        for epoch in range(Epocs):
            model.train()
            lossTotal = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].squeeze(1).to(device)
                attention_mask = batch["attention_mask"].squeeze(1).to(device)
                labels = batch["labels"].to(device)
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                lossTotal += loss.item()
            print(f"Epoch {epoch+1}/{Epocs} Loss: {lossTotal/len(train_loader):.4f}")

        model.eval()
        preds, true_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].squeeze(1).to(device)
                attention_mask = batch["attention_mask"].squeeze(1).to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        print(f"\nEvaluation Results - {bertModelInstance} {mode}")
        print("Accuracy:", accuracy_score(true_labels, preds))
        print(classification_report(true_labels, preds))

        print(f"Generating attention heatmap for {bertModelInstance} {mode}...")
        sample_text = X_test[0]
        inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, padding="max_length", max_length=40)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)
            if hasattr(output, "attentions") and output.attentions is not None:
                plot_attention_map(output.attentions, bertModelInstance, lora=Lora, tokenizer=tokenizer, sample_text=sample_text)
            else:
                print("No attention weights returned by model.")
