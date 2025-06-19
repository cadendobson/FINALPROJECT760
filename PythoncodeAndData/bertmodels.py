from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report
import torch
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
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Batchez = 8
Epocs = 3
RateLearn =  2e-5
loraVal = [4, 8, 16] 
LoraVal2 = [8, 16, 32]  
DropVals = [0.0, 0.1, 0.2]
Targets = [["query", "value"]]

HyperParams = list(product(loraVal, LoraVal2, DropVals, Targets))

for bertModelInstance in BertModels:
    for Lora in [False, True]:  
        mode = "with LoRA" if Lora else "without LoRA" 
        if not Lora:
            print(f"\nEvaluating {bertModelInstance} without LoRA >>>>>>>>>>\n")
            tokenizer = AutoTokenizer.from_pretrained(bertModelInstance)
            model = AutoModelForSequenceClassification.from_pretrained(bertModelInstance, num_labels=2)
            model.to(device)
        else:
            print(f"\nEvaluating {bertModelInstance} with LoRA hyperparameter tuning>>>>>>\n")
            for r, alpha, Dropout, targets in HyperParams:
                tokenizer = AutoTokenizer.from_pretrained(bertModelInstance)
                model = AutoModelForSequenceClassification.from_pretrained(bertModelInstance, num_labels=2)
                model.to(device)
                ConfigLora = LoraConfig(
                    r=r,
                    lora_alpha=alpha,
                    lora_dropout=Dropout,
                    target_modules=targets, 
                    bias="none",
                )
                model = get_peft_model(model, ConfigLora)

        TrainData = TextDataset(X_train, y_train, tokenizer)
        TestData = TextDataset(X_test, y_test, tokenizer)
        train_loader = DataLoader(TrainData, batch_size=Batchez, shuffle=True)
        test_loader = DataLoader(TestData, batch_size=Batchez)

        optimizer = torch.optim.AdamW(model.parameters(), lr=RateLearn)
        Trainingstepz = len(train_loader) * Epocs
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=Trainingstepz)

        for epoch in range(Epocs):
            print(f"\nStarting epoch {epoch + 1} for {bertModelInstance} {mode}...")
            model.train()
            lossTotal = 0
            for batch_idx, batch in enumerate(train_loader):
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
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                    print(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

            avg_loss = lossTotal / len(train_loader)
            print(f"Epoch {epoch + 1} completed for {bertModelInstance} {mode}. Average Loss: {avg_loss:.4f}")

        print(f"\nEvaluating {bertModelInstance} {mode} on the test set...")
        model.eval()
        preds, true_labels = [], []
        total_eval_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].squeeze(1).to(device)
                attention_mask = batch["attention_mask"].squeeze(1).to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                loss = outputs.loss
                total_eval_loss += loss.item()

                batch_preds = torch.argmax(logits, axis=1).cpu().numpy()
                preds.extend(batch_preds)
                true_labels.extend(labels.cpu().numpy())

        avg_eval_loss = total_eval_loss / len(test_loader)
        print(f"Test Loss: {avg_eval_loss:.4f}")
        print(f"Results for {bertModelInstance} {mode}:")
        print("Accuracy:", accuracy_score(true_labels, preds))
        print(classification_report(true_labels, preds))