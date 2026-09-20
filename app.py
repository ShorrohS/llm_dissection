import math
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import psutil
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "SmolLM2-135M"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="SmolLM2-135M Dissection Lab",
    description="Interactive Live LLM Architecture Visualizer & Deep Layer Inspector",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Initializing SmolLM2-135M on {DEVICE.upper()} with eager attention...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, attn_implementation="eager").to(DEVICE)
model.eval()

print("[+] SmolLM2-135M loaded successfully with eager attention.")

def get_system_memory() -> Dict[str, Any]:
    process = psutil.Process()
    ram_bytes = process.memory_info().rss
    vram_bytes = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    return {
        "ram_bytes": ram_bytes,
        "ram_mb": round(ram_bytes / (1024 * 1024), 2),
        "vram_bytes": vram_bytes,
        "vram_mb": round(vram_bytes / (1024 * 1024), 2),
        "device": DEVICE
    }

class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 60
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class SimulateRequest(BaseModel):
    prompt: str

@app.get("/info")
def get_info():
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "model_name": "SmolLM2-135M",
        "total_parameters": total_params,
        "total_weight_bytes_fp16": total_params * 2,
        "total_weight_mb_fp16": round((total_params * 2) / (1024 * 1024), 2),
        "total_weight_mb_fp32": round((total_params * 4) / (1024 * 1024), 2),
        "num_layers": len(model.model.layers),
        "hidden_size": model.config.hidden_size,
        "vocab_size": model.config.vocab_size,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": model.config.num_key_value_heads,
        "intermediate_size": model.config.intermediate_size,
        "device": DEVICE,
        "memory": get_system_memory()
    }

@app.get("/embedding/{token_id}")
def get_token_embedding(token_id: int):
    vocab_size = model.config.vocab_size
    if token_id < 0 or token_id >= vocab_size:
        raise HTTPException(status_code=400, detail=f"Token ID must be between 0 and {vocab_size - 1}.")

    with torch.no_grad():
        row = model.model.embed_tokens.weight[token_id].detach().cpu().float()
        token_str = tokenizer.decode([token_id])
        utf8_bytes = list(token_str.encode("utf-8"))

        return {
            "token_id": token_id,
            "token_str": token_str,
            "repr": repr(token_str),
            "byte_repr": utf8_bytes,
            "hex_repr": [f"0x{b:02X}" for b in utf8_bytes],
            "row_index": token_id,
            "tensor_name": "model.embed_tokens.weight",
            "safetensors_file": "model.safetensors",
            "total_dimensions": 576,
            "first_32": [round(float(v), 5) for v in row[:32].tolist()],
            "full_576": [round(float(v), 5) for v in row.tolist()],
            "mean": round(float(row.mean()), 5),
            "std": round(float(row.std()), 5),
            "min": round(float(row.min()), 5),
            "max": round(float(row.max()), 5)
        }

@app.post("/chat")
def chat(req: ChatRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    t0 = time.perf_counter()
    inputs = tokenizer(req.prompt, return_tensors="pt").to(DEVICE)
    input_length = inputs.input_ids.shape[1]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    do_sample = req.temperature > 0

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=max(req.temperature, 0.01) if do_sample else 1.0,
            top_p=req.top_p if do_sample else 1.0,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=do_sample,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id
        )

    t1 = time.perf_counter()
    new_tokens = output_ids[0][input_length:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    latency = max(t1 - t0, 0.001)

    return {
        "prompt": req.prompt,
        "response": generated_text,
        "tokens_prompt": input_length,
        "tokens_generated": len(new_tokens),
        "latency_sec": round(latency, 3),
        "tokens_per_sec": round(len(new_tokens) / latency, 2),
        "device": DEVICE,
        "memory": get_system_memory()
    }

@app.post("/simulate")
def simulate(req: SimulateRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    inputs = tokenizer(req.prompt, return_tensors="pt").to(DEVICE)
    input_ids_list = inputs.input_ids[0].tolist()

    # 1. Granular Token Breakdown with byte representation
    token_breakdown = []
    for idx, tid in enumerate(input_ids_list):
        t_str = tokenizer.decode([tid])
        token_breakdown.append({
            "index": idx,
            "id": int(tid),
            "token_str": t_str,
            "byte_repr": list(t_str.encode("utf-8")),
            "char_count": len(t_str),
            "hex_repr": [f"0x{b:02X}" for b in t_str.encode("utf-8")]
        })

    recorded_steps: List[Dict[str, Any]] = []
    hooks = []

    def make_hook(layer_id: str, layer_name: str, layer_type: str, layer_index: int, module: torch.nn.Module, is_head: bool = False):
        param_count = sum(p.numel() for p in module.parameters())
        mem_fp16 = param_count * 2
        mem_fp32 = param_count * 4

        def _hook(mod, args, output):
            t_out = output[0] if isinstance(output, tuple) else output
            in_shape = list(args[0].shape) if args and hasattr(args[0], "shape") else None
            out_shape = list(t_out.shape) if hasattr(t_out, "shape") else None

            token_vectors = []
            if t_out is not None and hasattr(t_out, "dim") and t_out.dim() >= 2:
                seq_len = t_out.shape[1]
                t_float = t_out[0].detach().cpu().float()

                for pos in range(seq_len):
                    vec = t_float[pos]
                    v_mean = float(vec.mean())
                    v_std = float(vec.std())
                    v_min = float(vec.min())
                    v_max = float(vec.max())

                    sample_16 = [round(float(val), 5) for val in vec[:16].tolist()]

                    if not is_head and vec.numel() <= 1024:
                        full_vec = [round(float(val), 4) for val in vec.tolist()]
                    else:
                        full_vec = sample_16

                    token_vectors.append({
                        "token_index": pos,
                        "token_id": token_breakdown[pos]["id"] if pos < len(token_breakdown) else None,
                        "token_str": token_breakdown[pos]["token_str"] if pos < len(token_breakdown) else "",
                        "mean": round(v_mean, 5),
                        "std": round(v_std, 5),
                        "min": round(v_min, 5),
                        "max": round(v_max, 5),
                        "sample_16": sample_16,
                        "vector_576": full_vec
                    })

            mem = get_system_memory()
            recorded_steps.append({
                "step_number": len(recorded_steps) + 1,
                "layer_id": layer_id,
                "layer_name": layer_name,
                "layer_type": layer_type,
                "layer_index": layer_index,
                "input_shape": in_shape,
                "output_shape": out_shape,
                "parameter_count": param_count,
                "memory_bytes_fp16": mem_fp16,
                "memory_mb_fp16": round(mem_fp16 / (1024 * 1024), 2),
                "memory_bytes_fp32": mem_fp32,
                "memory_mb_fp32": round(mem_fp32 / (1024 * 1024), 2),
                "ram_mb": mem["ram_mb"],
                "vram_mb": mem["vram_mb"],
                "token_vectors": token_vectors
            })

        return _hook

    hooks.append(
        model.model.embed_tokens.register_forward_hook(
            make_hook("embedding", "Embedding Layer", "Embedding(49152, 576)", 0, model.model.embed_tokens)
        )
    )

    for i, layer in enumerate(model.model.layers):
        hooks.append(
            layer.register_forward_hook(
                make_hook(f"block_{i}", f"Transformer Block {i}", "LlamaDecoderLayer", i + 1, layer)
            )
        )

    hooks.append(
        model.model.norm.register_forward_hook(
            make_hook("norm", "Final RMSNorm", "LlamaRMSNorm(576)", 31, model.model.norm)
        )
    )

    hooks.append(
        model.lm_head.register_forward_hook(
            make_hook("lm_head", "LM Head", "Linear(576, 49152)", 32, model.lm_head, is_head=True)
        )
    )

    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
            logits = outputs.logits
            # Extract real attention maps (averaged across 9 heads) from Block 0 and Block 29
            b0_matrix = [[round(float(v) * 100, 2) for v in row] for row in outputs.attentions[0][0].mean(dim=0).tolist()]
            b29_matrix = [[round(float(v) * 100, 2) for v in row] for row in outputs.attentions[-1][0].mean(dim=0).tolist()]
    finally:
        for h in hooks:
            h.remove()
    t1 = time.perf_counter()

    # Logit Lens: Early exit top-3 next-token predictions & top 3 mover dimensions per block
    # Computed safely after diagnostic hooks are removed
    logit_lens = []
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        with torch.no_grad():
            for i in range(30):
                curr_h = outputs.hidden_states[i + 1][:, -1, :]
                prev_h = outputs.hidden_states[i][:, -1, :]

                normed = model.model.norm(curr_h)
                l_i = model.lm_head(normed)[0]
                p_i = torch.softmax(l_i, dim=-1)
                top3 = torch.topk(p_i, 3)

                top_preds = []
                for r, (t_idx, prob_val) in enumerate(zip(top3.indices, top3.values)):
                    t_id = int(t_idx)
                    w = tokenizer.decode([t_id])
                    top_preds.append({
                        "rank": r + 1,
                        "token_id": t_id,
                        "token_str": w,
                        "prob": round(float(prob_val) * 100, 2)
                    })

                delta = (curr_h[0] - prev_h[0]).float()
                top3_dims = torch.topk(torch.abs(delta), 3).indices.tolist()
                movers = []
                for d in top3_dims:
                    movers.append({
                        "dim": int(d),
                        "delta": round(float(delta[d]), 3),
                        "val": round(float(curr_h[0, d]), 3)
                    })

                if i <= 9:
                    role = "Early Layer (Grammar & Syntax)"
                    role_desc = "Building local grammatical structure, token pairing, and fundamental word senses."
                elif i <= 21:
                    role = "Middle Layer (Knowledge & Facts)"
                    role_desc = "Associating factual knowledge, entity properties, and broad semantic themes."
                else:
                    role = "Late Layer (Decision & Token Choice)"
                    role_desc = "Collapsing contextual uncertainty into concrete next-word vocabulary probabilities."

                logit_lens.append({
                    "layer_idx": i,
                    "layer_name": f"Transformer Block {i}",
                    "role": role,
                    "role_desc": role_desc,
                    "top_predictions": top_preds,
                    "top_movers": movers
                })

    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    top_10 = torch.topk(last_logits, 10)

    candidate_list = []
    for rank, (tid_tensor, logit_tensor) in enumerate(zip(top_10.indices, top_10.values)):
        tid = int(tid_tensor)
        raw_l = float(logit_tensor)
        prob = float(probs[tid])
        t_str = tokenizer.decode([tid])
        candidate_list.append({
            "rank": rank + 1,
            "token_id": tid,
            "token_str": t_str,
            "repr": repr(t_str),
            "raw_logit": round(raw_l, 4),
            "calibrated_prob": round(prob * 100, 3)
        })

    return {
        "prompt": req.prompt,
        "token_count": len(input_ids_list),
        "token_breakdown": token_breakdown,
        "total_steps": len(recorded_steps),
        "layers": recorded_steps,
        "real_attentions": {
            "block_0": b0_matrix,
            "block_29": b29_matrix
        },
        "top_10_candidates": candidate_list,
        "predicted_token": candidate_list[0] if candidate_list else None,
        "logit_lens": logit_lens,
        "latency_sec": round(t1 - t0, 4),
        "system_memory": get_system_memory()
    }


class MicroMathRequest(BaseModel):
    prompt: Optional[str] = "Sun and Moon"
    selected_token_idx: Optional[int] = 0
    target_token_idx: Optional[int] = 1


@app.post("/micromath")
def get_micromath(req: MicroMathRequest):
    prompt = (req.prompt or "Sun and Moon").strip()
    if not prompt:
        prompt = "Sun and Moon"

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"][0].tolist()
    if not input_ids:
        raise HTTPException(status_code=400, detail="Unable to tokenize prompt.")

    tokens = []
    for i, tid in enumerate(input_ids):
        t_str = tokenizer.decode([tid])
        tokens.append({
            "idx": i,
            "id": tid,
            "str": t_str,
            "repr": repr(t_str)
        })

    sel_idx = max(0, min(req.selected_token_idx if req.selected_token_idx is not None else 0, len(tokens) - 1))
    # Target token for attention score: prefer a different token if available
    default_tgt = 1 if len(tokens) > 1 else 0
    tgt_idx = req.target_token_idx if req.target_token_idx is not None else default_tgt
    tgt_idx = max(0, min(tgt_idx, len(tokens) - 1))

    with torch.no_grad():
        embeds = model.model.embed_tokens(inputs["input_ids"])[0]  # [seq_len, 576]
        
        # 1. 3D slices for all tokens
        tokens_3d = []
        for i, t in enumerate(tokens):
            v3 = embeds[i, :3].detach().cpu().float().tolist()
            tokens_3d.append({
                **t,
                "coords": [round(float(x), 3) for x in v3]
            })

        # 2. Step A: Real 3x3 weight submatrices from Block 0
        q_weight_full = model.model.layers[0].self_attn.q_proj.weight.detach().cpu().float()
        k_weight_full = model.model.layers[0].self_attn.k_proj.weight.detach().cpu().float()

        w_q_3x3 = [[round(float(val), 3) for val in row] for row in q_weight_full[:3, :3].tolist()]
        w_k_3x3 = [[round(float(val), 3) for val in row] for row in k_weight_full[:3, :3].tolist()]

        # 3. Step B: Query Multiplication for selected token
        sel_x = [float(x) for x in tokens_3d[sel_idx]["coords"]]
        q_rows = []
        q_vec = []
        for row_i in range(3):
            terms = []
            row_sum = 0.0
            for col_j in range(3):
                xj = sel_x[col_j]
                wij = w_q_3x3[row_i][col_j]
                prod = xj * wij
                row_sum += prod
                terms.append({
                    "x_idx": col_j,
                    "x_val": xj,
                    "w_val": wij,
                    "prod": round(prod, 3)
                })
            sum_rounded = round(row_sum, 3)
            q_vec.append(sum_rounded)
            eq_str = " + ".join([f"({t['x_val']} × {t['w_val']})" for t in terms]) + f" = {sum_rounded}"
            q_rows.append({
                "row_idx": row_i,
                "terms": terms,
                "sum": sum_rounded,
                "equation": eq_str
            })

        # 4. Step C: Key projection for target token and Attention Score
        tgt_x = [float(x) for x in tokens_3d[tgt_idx]["coords"]]
        k_rows = []
        k_vec = []
        for row_i in range(3):
            terms = []
            row_sum = 0.0
            for col_j in range(3):
                xj = tgt_x[col_j]
                wkij = w_k_3x3[row_i][col_j]
                prod = xj * wkij
                row_sum += prod
                terms.append({
                    "x_idx": col_j,
                    "x_val": xj,
                    "w_val": wkij,
                    "prod": round(prod, 3)
                })
            sum_rounded = round(row_sum, 3)
            k_vec.append(sum_rounded)
            eq_str = " + ".join([f"({t['x_val']} × {t['w_val']})" for t in terms]) + f" = {sum_rounded}"
            k_rows.append({
                "row_idx": row_i,
                "terms": terms,
                "sum": sum_rounded,
                "equation": eq_str
            })

        # Dot product: q_vec . k_vec
        score_terms = []
        raw_score = 0.0
        for dim_i in range(3):
            prod = q_vec[dim_i] * k_vec[dim_i]
            raw_score += prod
            score_terms.append({
                "dim": dim_i,
                "q_val": q_vec[dim_i],
                "k_val": k_vec[dim_i],
                "prod": round(prod, 4)
            })
        raw_score = round(raw_score, 4)
        score_eq = " + ".join([f"({t['q_val']} × {t['k_val']})" for t in score_terms]) + f" = {raw_score}"

        # 5. Step D: MLP Filter & SiLU Activation
        # Take representative values: q_vec coords plus a guaranteed negative demonstrator
        demo_vals = list(q_vec)
        # If none of q_vec is negative, append a coordinate from embedding or a fixed negative example to guarantee visual filter demonstration
        if not any(v < -0.05 for v in demo_vals):
            # Check embedding coords for negative value
            found_neg = None
            for c in sel_x:
                if c < -0.05:
                    found_neg = c
                    break
            demo_vals.append(found_neg if found_neg is not None else -1.15)

        silu_results = []
        for d_idx, val in enumerate(demo_vals[:4]):
            z = float(val)
            sig = 1.0 / (1.0 + math.exp(-z))
            silu_z = z * sig
            is_compressed = (z < 0)
            shrink_pct = round(abs((silu_z - z) / (abs(z) + 1e-9)) * 100, 1)
            
            silu_results.append({
                "dim_label": f"Dim #{d_idx}" if d_idx < 3 else "Sample Coordinate",
                "raw_z": round(z, 3),
                "sigmoid": round(sig, 3),
                "silu_z": round(silu_z, 3),
                "is_compressed": is_compressed,
                "shrink_pct": shrink_pct,
                "explanation": "The model decides this specific number isn't useful for the next layer, so it shrinks it." if is_compressed else "The model finds this concept relevant, so it passes it through intact."
            })

    return {
        "prompt": prompt,
        "tokens": tokens_3d,
        "selected_token_idx": sel_idx,
        "target_token_idx": tgt_idx,
        "step_a_weights": {
            "matrix_name": "model.layers.0.self_attn.q_proj.weight[:3, :3]",
            "w_q_3x3": w_q_3x3,
            "w_k_3x3": w_k_3x3
        },
        "step_b_multiplication": {
            "token": tokens_3d[sel_idx],
            "input_x": sel_x,
            "equations": q_rows,
            "output_q": q_vec
        },
        "step_c_score": {
            "query_token": tokens_3d[sel_idx],
            "key_token": tokens_3d[tgt_idx],
            "query_vec": q_vec,
            "key_vec": k_vec,
            "score_terms": score_terms,
            "raw_score": raw_score,
            "equation": score_eq
        },
        "step_d_filter": {
            "formula": "SiLU(z) = z * sigmoid(z) = z / (1 + e^(-z))",
            "items": silu_results
        }
    }


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(index_file)
