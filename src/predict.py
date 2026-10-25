import sys
from transformers import pipeline

MODEL_DIR = "outputs/model"

classifier = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR)

def predict(text: str) -> dict:
    result = classifier(text, truncation=True, max_length=256)[0]
    return {"label": result["label"], "confidence": round(result["score"], 4)}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        samples = [" ".join(sys.argv[1:])]
    else:
        samples = [
            "This movie was absolutely fantastic, I loved every minute!",
            "Terrible plot, bad acting, a complete waste of time.",
            "It was okay, nothing special but not awful either.",
        ]
    for s in samples:
        print(f"{s}\n  -> {predict(s)}\n")