import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "outputs/model"
TEST_SIZE = 500
MAX_LENGTH = 256
BATCH_SIZE = 16

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Same test subset as in training (same seed)
test_ds = load_dataset("stanfordnlp/imdb")["test"].shuffle(seed=42).select(range(TEST_SIZE))
texts = test_ds["text"]
labels = np.array(test_ds["label"])

preds = []
with torch.no_grad():
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch, truncation=True, max_length=MAX_LENGTH,
            padding=True, return_tensors="pt",
        )
        logits = model(**enc).logits
        preds.extend(torch.argmax(logits, dim=-1).tolist())
        print(f"Processed {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

preds = np.array(preds)

names = ["NEGATIVE", "POSITIVE"]
print("\n" + classification_report(labels, preds, target_names=names, digits=3))

cm = confusion_matrix(labels, preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - DistilBERT on IMDB")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")