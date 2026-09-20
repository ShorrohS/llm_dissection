import csv
from pathlib import Path
from safetensors import safe_open

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "SmolLM2-135M" / "model.safetensors"
OUTPUT_CSV = BASE_DIR / "embedding_sample.csv"

def export_embedding_to_csv(tensor_name: str = "model.embed_tokens.weight", sample_size: int = 1000):
    print(f"Opening {MODEL_PATH}...")
    with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"Found {len(keys)} tensors in file.")

        if tensor_name not in keys:
            matching = [k for k in keys if "embed" in k]
            tensor_name = matching[0] if matching else keys[0]
            print(f"Using '{tensor_name}'.")

        print(f"Extracting tensor '{tensor_name}'...")
        tensor_data = f.get_tensor(tensor_name)
        print(f"Full tensor shape: {tensor_data.shape} (dtype: {tensor_data.dtype})")

        # Slice the sample rows and convert to numpy array
        sample = tensor_data[:sample_size].float().numpy()
        num_rows, num_cols = sample.shape

        print(f"Saving {num_rows} rows x {num_cols} columns to {OUTPUT_CSV.name}...")
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.writer(f_out)
            # Column headers: 0, 1, 2, ... (matching DataFrame default behavior)
            writer.writerow([str(i) for i in range(num_cols)])
            writer.writerows(sample.tolist())

        print("\n" + "=" * 60)
        print("CSV export completed successfully!")
        print(f"Output CSV path: {OUTPUT_CSV}")
        print(f"Dimensions: {num_rows} rows x {num_cols} columns")
        print("=" * 60)

if __name__ == "__main__":
    export_embedding_to_csv()
