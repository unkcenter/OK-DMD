"""
Multidomain Stress Test Benchmark for OK-DMD vs. H2O KV-Cache Eviction.

This script simulates a severe context-shift scenario:
Astronomy -> Lasagna Recipe -> Astronomy Recall
It measures how well each eviction strategy retains older, globally important 
attractor states when subjected to temporary out-of-domain noise.
"""

import os
import sys
import torch
import numpy as np
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

# Dynamic path injection to allow seamless local package imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ok_dmd.eviction import OKDMDEviction

# Configure logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Setup reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Benchmark] Initializing execution on: {device}")

# ==============================================================================
# 1. ORGANIZE DATASET: STRESS TEST PROMPT GENERATION
# ==============================================================================
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"[Benchmark] Loading foundational model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)

# Domain A (Primary Context)
text_solar = """The Solar System is the gravitationally bound system of the Sun and the objects that orbit it. 
It formed 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud. 
The vast majority of the system's mass is in the Sun, with the majority of the remaining mass contained in Jupiter. 
The four inner system planets—Mercury, Venus, Earth, and Mars—are terrestrial planets, composed of rock. 
The four outer system planets are giant planets, substantially more massive than the terrestrials. 
The two largest, Jupiter and Saturn, are gas giants, composed mainly of hydrogen and helium.""" * 3

# Domain B (Transient Distraction Noise)
text_cooking = """ To prepare a traditional Italian lasagna, you must master the bolognese sauce, which requires hours 
of slow simmering with beef, pork, tomatoes, onions, carrots, and a splash of red wine. Layering is crucial: 
alternate sheets of fresh egg pasta, rich meat sauce, and smooth, velvety béchamel sauce. Top the final layer 
generously with grated Parmigiano-Reggiano cheese. Bake in a preheated oven at 180 degrees Celsius.""" * 3

# Domain A' (Recall Phase)
text_recall = """ Now, returning strictly to the previous topic about astronomy and space exploration: the planets 
in the outer Solar System, specifically the gas giants Jupiter and Saturn, orbit the Sun. When evaluating the 
gravitational binding and mass distribution of these terrestrial and giant celestial bodies within the disc...""" * 2

long_text = text_solar + text_cooking + text_recall
inputs = tokenizer(long_text, return_tensors="pt").to(device)
total_tokens = inputs["input_ids"].shape[1]
print(f"[Benchmark] Tokenized prompt contains: {total_tokens} tokens")

# Extract ground-truth key embeddings from Layer 10, Head 0
with torch.no_grad():
    outputs = model(**inputs, use_cache=True)
past_keys = outputs.past_key_values

layer_idx = 10
head_idx = 0
extracted_keys = None

# Adaptive Key Extraction Block (supports v4 and v5 HF structures)
if hasattr(past_keys, "layers") and len(past_keys.layers) > layer_idx:
    try: extracted_keys = past_keys.layers[layer_idx].keys[0, head_idx].cpu()
    except Exception: pass
if extracted_keys is None and hasattr(past_keys, "key_cache") and len(past_keys.key_cache) > layer_idx:
    try: extracted_keys = past_keys.key_cache[layer_idx][0, head_idx].cpu()
    except Exception: pass
if extracted_keys is None:
    try: extracted_keys = past_keys[layer_idx][0][0, head_idx].cpu()
    except Exception: pass

if extracted_keys is None:
    raise TypeError("Critical Error: Failed to extract key embeddings from cache.")

head_dim = extracted_keys.shape[-1]
print(f"[Benchmark] Key embeddings extracted. Shape: {extracted_keys.shape}")

# ==============================================================================
# 2. EXPERIMENT PARAMETERS
# ==============================================================================
N_prefill = int(total_tokens * 0.55)  # Covers primary domain and start of distraction
N_decode = total_tokens - N_prefill - 1
target_budget = 64    # Aggressive 92.2% compression budget
sink_size = 4
window_size = 16

# Split sequence into historical context and evaluation queries
keys_history = extracted_keys[:N_prefill + N_decode]
queries = keys_history[N_prefill:]

# Initialize H2O states
h2o_cache_indices = list(range(N_prefill))
h2o_cumulative_attn = torch.zeros(N_prefill + N_decode)

# Initialize OK-DMD states
dmd_cache_indices = list(range(N_prefill))
lambda_ = 0.995
sigma = 1e-5
k_modes = 16
update_interval = 16

# ==============================================================================
# 3. CONVERGE OK-DMD SUBSPACE (PREFILL WARM-UP)
# ==============================================================================
print(f"\n[OK-DMD] Initializing Koopman state and converging subspace...")
evictor = OKDMDEviction(
    num_heads_kv=1,
    head_dim=head_dim,
    k_modes=k_modes,
    lambda_forget=lambda_,
    stab_reg=sigma,
    sink_size=sink_size,
    window_size=window_size,
    update_interval=update_interval
)

# Initialize on the matching device and float32 dtype
evictor.initialize(batch_size=1, device=device, dtype=torch.float32)

# Pass parallel prefill block to build the initial dynamic model
prefill_keys_tensor = keys_history[:N_prefill].view(1, 1, N_prefill, head_dim).to(device)
evictor.prefill_step(prefill_keys_tensor, qr_iterations=10)
print(f"[OK-DMD] Subspace converged. Running sequential simulation...\n")

# ==============================================================================
# 4. SIMULATION DECODE LOOP (OK-DMD vs. H2O)
# ==============================================================================
h2o_similarity = []
dmd_similarity = []

for step in range(N_decode):
    t_global = N_prefill + step
    q_t = queries[step].to(device)
    k_t = keys_history[t_global].to(device)
    
    # --- GROUND TRUTH ATTENTION ---
    full_keys = keys_history[:t_global + 1].to(device)
    attn_full_logits = torch.matmul(full_keys, q_t) / np.sqrt(head_dim)
    attn_full = torch.softmax(attn_full_logits, dim=-1)
    
    # --------------------------------------------------------------------------
    # BASELINE: H2O Eviction
    # --------------------------------------------------------------------------
    h2o_cache_indices.append(t_global)
    h2o_keys = keys_history[h2o_cache_indices].to(device)
    attn_h2o_step = torch.softmax(torch.matmul(h2o_keys, q_t) / np.sqrt(head_dim), dim=-1)
    
    for idx, weight in zip(h2o_cache_indices, attn_h2o_step):
        h2o_cumulative_attn[idx] += weight.item()
        
    if len(h2o_cache_indices) > target_budget:
        protected = set(h2o_cache_indices[:sink_size] + h2o_cache_indices[-window_size:])
        candidates = [idx for idx in h2o_cache_indices if idx not in protected]
        num_to_drop = len(h2o_cache_indices) - target_budget
        
        candidates.sort(key=lambda idx: h2o_cumulative_attn[idx].item())
        to_drop = set(candidates[:num_to_drop])
        h2o_cache_indices = [idx for idx in h2o_cache_indices if idx not in to_drop]
        
    attn_h2o_full_dim = torch.zeros(t_global + 1, device=device)
    h2o_keys_active = keys_history[h2o_cache_indices].to(device)
    weights_h2o = torch.softmax(torch.matmul(h2o_keys_active, q_t) / np.sqrt(head_dim), dim=-1)
    for idx, w in zip(h2o_cache_indices, weights_h2o):
        attn_h2o_full_dim[idx] = w

    # --------------------------------------------------------------------------
    # OUR PROPOSAL: OK-DMD Eviction
    # --------------------------------------------------------------------------
    # Format inputs for OKDMDEviction class
    k_prev_batch = keys_history[t_global - 1].view(1, 1, head_dim).to(device)
    k_curr_batch = k_t.view(1, 1, head_dim).to(device)
    
    # Step 1: Update Koopman model recursively
    evictor.decode_step(k_prev_batch, k_curr_batch)
    
    dmd_cache_indices.append(t_global)
    
    # Step 2: Evaluate and Evict
    if len(dmd_cache_indices) > target_budget:
        K_cache_batch = keys_history[dmd_cache_indices].view(1, 1, len(dmd_cache_indices), head_dim).to(device)
        scores = evictor.compute_eviction_scores(K_cache_batch) # [1, 1, cache_size]
        
        # Generate eviction mask
        mask = evictor.get_eviction_mask(scores, target_budget) # [1, 1, cache_size]
        mask_1d = mask[0, 0] # Extract batch and head dimensions
        
        # Filter indices based on OK-DMD decisions
        dmd_cache_indices = [idx for i, idx in enumerate(dmd_cache_indices) if mask_1d[i]]

    attn_dmd_full_dim = torch.zeros(t_global + 1, device=device)
    dmd_keys_active = keys_history[dmd_cache_indices].to(device)
    weights_dmd = torch.softmax(torch.matmul(dmd_keys_active, q_t) / np.sqrt(head_dim), dim=-1)
    for idx, w in zip(dmd_cache_indices, weights_dmd):
        attn_dmd_full_dim[idx] = w

    # --------------------------------------------------------------------------
    # METRIC CALCULATION: Cosine Similarity
    # --------------------------------------------------------------------------
    cos_sim_h2o = torch.nn.functional.cosine_similarity(attn_full.unsqueeze(0), attn_h2o_full_dim.unsqueeze(0)).item()
    cos_sim_dmd = torch.nn.functional.cosine_similarity(attn_full.unsqueeze(0), attn_dmd_full_dim.unsqueeze(0)).item()
    
    h2o_similarity.append(cos_sim_h2o)
    dmd_similarity.append(cos_sim_dmd)

# ==============================================================================
# 5. METRIC REPORTING
# ==============================================================================
h2o_mean = np.mean(h2o_similarity)
dmd_mean = np.mean(dmd_similarity)

print("-" * 60)
print(" FINAL PERFORMANCE REPORT: DOMAIN SHIFT STRESS TEST")
print(f" Total Sequence Length:    {N_prefill + N_decode} tokens")
print(f" KV-Cache Budget Limit:    {target_budget} tokens (Aggressive Compression)")
print("-" * 60)
print(f" -> Mean Attention Coherence (H2O Baseline): {h2o_mean:.6f}")
print(f" -> Mean Attention Coherence (OK-DMD Ours):  {dmd_mean:.6f}")
print("-" * 60)

if dmd_mean > h2o_mean:
    print("[SUCCESS] OK-DMD outperformed H2O! The dynamic attractor preserved the old Astronomy context.")
else:
    print("[ANALYSIS] H2O maintained a local advantage. Consider adjusting update_interval or target_budget.")
