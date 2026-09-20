import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
LOCAL_DIR = Path(__file__).resolve().parent / "models" / "SmolLM2-135M"

def download():
    print(f"Downloading model and tokenizer from '{MODEL_ID}'...")
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(LOCAL_DIR)

    print("Fetching model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.save_pretrained(LOCAL_DIR)

    print("\n" + "=" * 60)
    print("Download completed successfully!")
    print(f"Local directory path: {LOCAL_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    download()
