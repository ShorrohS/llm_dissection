# 🔬 SmolLM2-135M Dissection Lab

An interactive, beginner-friendly LLM architecture visualizer, mathematical dissection dashboard, and dual-mode playground built on **FastAPI**, **PyTorch**, and a **Japanese Zen Minimalist UI**.

Designed to explain how modern Large Language Models work from first principles: from raw characters to token IDs, high-dimensional vector embeddings in $\mathbb{R}^{576}$, multi-head self-attention mixing across 30 transformer layers, and calibrated probability predictions.

---

## ✨ Features

### 1. 🌊 Token Flow & Mathematical Simulator (Tab 1)
- **Granular Tokenization Breakdown**: View exact token IDs, string slices, character counts, and hexadecimal UTF-8 bytes (e.g. `0x20` for leading spaces).
- **Direct Safetensors Memory Lookup**: Click any token to inspect its direct row coordinate lookup inside `model.safetensors` (`model.embed_tokens.weight`).
- **33-Stage Pipeline Spine**: Watch activations ripple through the Embedding Layer, 30 Transformer Blocks, Final RMSNorm, and the LM Head.
- **Top 10 Softmax Predictions**: Real-time next-token probability distribution with raw logits and calibrated percentages.

### 2. 🔍 Layer Deep-Dive & Tensor Inspector (Tab 2)
- **Layer Hierarchy Tree**: Browse all 33 transformation checkpoints with exact parameter counts and FP16/FP32 memory footprints.
- **Interactive 576-Dimension Vector Heatmap**: A $24 \times 24$ coordinate cell grid color-coded from Terracotta (negative) to Washi (neutral) to Matcha (positive). Hover over any dimension to reveal its exact index (`0` to `575`) and float activation value.
- **Plain-English Explanations**: Accessible breakdowns of what every layer is accomplishing mathematically.

### 3. 💬 Conversational Playground (Tab 3)
- Minimalist chat interface powered by `model.generate()`.
- **Anti-Looping Engine**: Equipped with `repetition_penalty=1.2` and `no_repeat_ngram_size=3` for clean, fluent responses.
- Real-time generation telemetry: Latency, token generation throughput (`tok/s`), and RAM footprint.

### 4. 📖 First-Principles Interactive Card Deck (Tab 4)
Step-by-step interactive cards breaking down core concepts:
1. **Words to Numbers**: Byte-Pair Encoding and vocabulary tables ($0 \dots 49151$).
2. **Numbers to Coordinates**: Safetensors row lookup into 576 continuous dimensions.
3. **Words Talking to Words (Interactive Self-Attention)**:
   - Click words in an interactive 5-word sentence (`"The river bank was silent"`).
   - Arched connecting SVG arrows show dynamic **Attention Scores** ($Q \times K^\top / \sqrt{d_k}$) with opacity mapped to connection strength.
   - Live 576-dimension vector absorption wave animation below the words.
4. **Picking the Winner**: LM Head dot products and Softmax probability calibration.

---

## 🏗️ Architecture Specifications (SmolLM2-135M)

| Specification | Value |
| :--- | :--- |
| **Parameters** | 134,515,008 (~135M) |
| **Hidden Dimension ($d_{\text{model}}$)** | 576 |
| **Transformer Layers** | 30 blocks (`LlamaDecoderLayer`) |
| **Attention Heads** | 9 query heads, 3 key/value heads (GQA) |
| **Intermediate Size (MLP)** | 1,536 (SwiGLU) |
| **Vocabulary Size** | 49,152 tokens |
| **FP16 Weight Footprint** | ~269.03 MB |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/ShorrohS/llm_dissection.git
cd llm_dissection
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download SmolLM2-135M model weights
```bash
python download_model.py
```

### 5. Launch the application
```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser.

---

## 🛠️ Project Structure

```text
llm_dissection/
├── app.py                  # FastAPI server with forward hooks, /chat, /simulate, /embedding
├── download_model.py       # Script to download SmolLM2-135M from Hugging Face
├── map_embeddings.py       # Extracts embedding weights mapped to token strings
├── export_to_csv.py        # Safetensors weight exporter
├── requirements.txt        # Python package dependencies
├── .gitignore              # Ignores large binary weights and venvs
├── static/
│   └── index.html          # Japanese Zen Minimalist 4-tab SPA with interactive SVG visualizer
└── models/
    └── SmolLM2-135M/       # Local model configuration & tokenizer files
```

---

## 📜 License

MIT License &copy; 2026 ShorrohS
