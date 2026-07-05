"""
Stress Test Benchmark: Multi-Domain Context Shift (Astronomy -> Cooking -> Recall).

This benchmark evaluates the resilience of OK-DMD against attention drift
by introducing out-of-domain distraction noise and measuring attention coherence
after a returning prompt recall phase.
"""

import os
import sys
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# Environment-aware path resolution for standard package import
try:
    current_file = __file__
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(current_file)))
    sys.path.append(parent_dir)
except NameError:
    # Fallback for interactive/notebook execution
    sys.path.append(os.getcwd())

from ok_dmd.eviction import OKDMDEviction

# Disable verbose HF warning logs
logging.getLogger("transformers").setLevel(logging.ERROR)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Benchmark] Target device: {device}")

# ==============================================================================
# 1. DATASET PREPARATION: MULTI-DOMAIN PROMPT CONCATENATION
# ==============================================================================
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"[Benchmark] Loading model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)

# Primary Domain (Domain A)
text_solar = """The Solar System is the gravitationally bound system of the Sun and the objects that orbit it. 
It formed 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud. 
The vast majority of the system's mass is in the Sun, with the majority of the remaining mass contained in Jupiter. 
The four inner system planets—Mercury, Venus, Earth, and Mars—are terrestrial planets, composed of rock. 
The four outer system planets are giant planets, substantially more massive than the terrestrials. 
The two largest, Jupiter and Saturn, are gas giants, composed mainly of hydrogen and helium.""" * 3

# Out-of-Domain Distraction (Domain B)
text_cooking = """ To prepare a traditional Italian lasagna, you must master the bolognese sauce, which requires hours 
of slow simmering with beef, pork, tomatoes, onions, carrots, and a splash of red wine. Layering is crucial: 
alternate sheets of fresh egg pasta, rich meat sauce, and smooth, velvety béchamel sauce. Top the final layer 
generously with grated Parmigiano-Reggiano cheese. Bake in a preheated oven at 180 degrees Celsius.""" * 3

# In-Domain Recall Phase (Domain A')
text_recall = """ Now, returning strictly to the previous topic about astronomy and space exploration: the planets 
in the outer Solar System, specifically the gas giants Jupiter and Saturn, orbit the Sun. When evaluating the 
gravitational binding and mass distribution of these terrestrial and giant celestial bodies within the disc...""" * 2

long_text = text_solar + text_cooking + text_recall
inputs = tokenizer(long_text, return_tensors="pt").to(device)
total_tokens = inputs["input_ids"].shape[1]
print(f"[Benchmark] Prompt size: {total_tokens} tokens")

# Extract historical keys
with torch.no_grad():
    outputs = model(**inputs, use_cache=True)
past_keys = outputs.past_key_values

layer_idx = 10
head_idx = 0
extracted_keys = None

# Cascade extraction supporting multiple HF cache formats
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
    raise TypeError("Failed to extract keys from model cache.")

head_dim = extracted_keys.shape[-1]
print(f"[Benchmark] Extracted key tensor shape: {extracted_keys.shape}")

# ==============================================================================
# 2. EVALUATION CONFIGURATIONS
# ==============================================================================
N_prefill = int(total_tokens * 0.55)
N_decode = total_tokens - N_prefill - 1
target_budget = 64
sink_size = 4
window_size = 16

keys_history = extracted_keys[:N_prefill + N_decode]
queries = keys_history[N_prefill:]

# Base states
h2o_cache_indices = list(range(N_prefill))
h2o_cumulative_attn = torch.zeros(N_prefill + N_decode)

dmd_cache_indices = list(range(N_prefill))
lambda_ = 0.995
sigma = 1e-5
k_modes = 16
update_interval = 16

# ==============================================================================
# 3. PROMPT WARM-UP & SUBSPACE CONVERGENCE
# ==============================================================================
print("[OK-DMD] Initializing states and converging projection subspace...")
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
evictor.initialize(batch_size=1, device=device, dtype=torch.float32)

prefill_keys = keys_history[:N_prefill].view(1, 1, N_prefill, head_dim).to(device)
evictor.prefill_step(prefill_keys, qr_iterations=10)
print("[OK-DMD] Subspace converged. Simulating sequential decode phase...\n")

# ==============================================================================
# 4. SEQUENTIAL DECODE SIMULATION
# ==============================================================================
h2o_similarity = []
dmd_similarity = []

for step in range(N_decode):
    t_global = N_prefill + step
    q_t = queries[step].to(device)
    k_t = keys_history[t_global].to(device)
    
    # Ground truth causal attention over full historical context
    full_keys = keys_history[:t_global + 1].to(device)
    attn_full = torch.softmax(torch.matmul(full_keys, q_t) / np.sqrt(head_dim), dim=-1)
    
    # --- H2O Baseline ---
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

    # --- OK-DMD Ours ---
    k_prev = keys_history[t_global - 1].view(1, 1, head_dim).to(device)
    k_curr = k_t.view(1, 1, head_dim).to(device)
    
    evictor.decode_step(k_prev, k_curr)
    dmd_cache_indices.append(t_global)
    
    if len(dmd_cache_indices) > target_budget:
        K_cache_batch = keys_history[dmd_cache_indices].view(1, 1, len(dmd_cache_indices), head_dim).to(device)
        scores = evictor.compute_eviction_scores(K_cache_batch)
        
        mask = evictor.get_eviction_mask(scores, target_budget)
        mask_1d = mask[0, 0]
        
        dmd_cache_indices = [idx for i, idx in enumerate(dmd_cache_indices) if mask_1d[i]]

    attn_dmd_full_dim = torch.zeros(t_global + 1, device=device)
    dmd_keys_active = keys_history[dmd_cache_indices].to(device)
    weights_dmd = torch.softmax(torch.matmul(dmd_keys_active, q_t) / np.sqrt(head_dim), dim=-1)
    for idx, w in zip(dmd_cache_indices, weights_dmd):
        attn_dmd_full_dim[idx] = w

    # Compute metric
    cos_sim_h2o = torch.nn.functional.cosine_similarity(attn_full.unsqueeze(0), attn_h2o_full_dim.unsqueeze(0)).item()
    cos_sim_dmd = torch.nn.functional.cosine_similarity(attn_full.unsqueeze(0), attn_dmd_full_dim.unsqueeze(0)).item()
    
    h2o_similarity.append(cos_sim_h2o)
    dmd_similarity.append(cos_sim_dmd)

# ==============================================================================
# 5. SCIENTIFIC PLOT GENERATION
# ==============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['text.color'] = '#1E293B'
plt.rcParams['axes.labelcolor'] = '#1E293B'
plt.rcParams['xtick.color'] = '#475569'
plt.rcParams['ytick.color'] = '#475569'

fig, ax = plt.subplots(figsize=(11, 5), dpi=300)

color_h2o = '#EF4444' # Crimson
color_dmd = '#0D9488' # Deep Teal

ax.plot(h2o_similarity, label="Baseline SOTA: H2O (Cumulative Attention)", color=color_h2o, linestyle="--", linewidth=1.5, alpha=0.8)
ax.plot(dmd_similarity, label="Ours: OK-DMD (Koopman Attractor Subspace)", color=color_dmd, linewidth=2.0)

# Compute sequence shift indexes
len_solar = len(tokenizer(text_solar, add_special_tokens=False)["input_ids"])
len_cooking = len(tokenizer(text_cooking, add_special_tokens=False)["input_ids"])
transition_step = (len_solar + len_cooking) - N_prefill

if 0 < transition_step < N_decode:
    # Distraction shade
    ax.axvspan(0, transition_step, color='#FFF1F2', alpha=0.6, label='Lasagna Distraction (Out of Domain)')
    ax.text(transition_step / 2, 0.72, 'Distraction Phase\n(Lasagna Recipe, OOD)', color='#9F1239', fontsize=9, ha='center', weight='bold')
    
    # Recall shade
    ax.axvspan(transition_step, N_decode - 1, color='#F0FDF4', alpha=0.6, label='Astronomy Recall (In-Domain)')
    ax.text(transition_step + (N_decode - transition_step) / 2, 0.72, 'Recall Phase\n(Return to Space)', color='#166534', fontsize=9, ha='center', weight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CBD5E1')
ax.spines['bottom'].set_color('#CBD5E1')
ax.grid(True, linestyle=":", alpha=0.5, color='#94A3B8')

ax.set_title("Multi-Domain Stress Test: H2O vs. OK-DMD\nAttention Coherence under 92.2% KV-Cache Compression Budget", fontsize=12, weight='bold', pad=15)
ax.set_xlabel("Decode Step (Autoregressive Generation)", fontsize=10, labelpad=10)
ax.set_ylabel("Attention Cosine Similarity (vs. Full Cache)", fontsize=10, labelpad=10)

min_val = min(min(h2o_similarity), min(dmd_similarity))
ax.set_ylim(max(0.0, min_val - 0.08), 1.05)
ax.set_xlim(0, N_decode - 1)

ax.legend(loc="lower left", frameon=True, facecolor='white', edgecolor='none', fontsize=9)
plt.tight_layout()

os.makedirs("assets", exist_ok=True)
plt.savefig("assets/benchmark_results.png", dpi=300)
print("[Benchmark] High-resolution plot saved successfully to 'assets/benchmark_results.png'")
plt.show()

# Final reporting
h2o_mean = np.mean(h2o_similarity)
dmd_mean = np.mean(dmd_similarity)
print("-" * 60)
print(f" -> Mean Attention Coherence (H2O):      {h2o_mean:.6f}")
print(f" -> Mean Attention Coherence (OK-DMD):   {dmd_mean:.6f}")
print("-" * 60)
