from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer
from safetensors import safe_open

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "SmolLM2-135M"
OUTPUT_CSV = BASE_DIR / "mapped_embeddings.csv"
SAMPLE_SIZE = 1000

def main():
    print(f"Loading tokenizer from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    weights_file = MODEL_DIR / "model.safetensors"
    print(f"Loading embedding weights from {weights_file}...")
    with safe_open(weights_file, framework="pt", device="cpu") as f:
        tensor_data = f.get_tensor("model.embed_tokens.weight")
    
    print(f"Embedding tensor shape: {tensor_data.shape} (dtype: {tensor_data.dtype})")

    # Slice the first SAMPLE_SIZE rows
    embeddings_sample = tensor_data[:SAMPLE_SIZE].float().numpy()
    num_rows, num_dims = embeddings_sample.shape

    print(f"Generating Token IDs and decoding strings for first {num_rows} tokens...")
    token_ids = list(range(num_rows))
    token_strings = [tokenizer.decode([i]) for i in token_ids]

    # Construct DataFrame
    print("Constructing Pandas DataFrame...")
    embed_cols = [f"dim_{i}" for i in range(num_dims)]
    embed_df = pd.DataFrame(embeddings_sample, columns=embed_cols)

    df = pd.DataFrame({
        "Token ID": token_ids,
        "Token String": token_strings,
    })
    df = pd.concat([df, embed_df], axis=1)

    print(f"Saving DataFrame ({df.shape[0]} rows x {df.shape[1]} cols) to {OUTPUT_CSV.name}...")
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("\n" + "=" * 60)
    print("Mapped embeddings saved successfully!")
    print(f"File: {OUTPUT_CSV}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("First 5 rows preview:")
    print(df[["Token ID", "Token String", "dim_0", "dim_1", "dim_2"]].head())
    print("=" * 60)

if __name__ == "__main__":
    main()
