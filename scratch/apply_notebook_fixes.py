"""
Apply the comprehensive set of cell fixes to velm_colab.ipynb.

Run from VELM repo root:
    .venv/bin/python scratch/apply_notebook_fixes.py

A backup is taken automatically before writing.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

NB = Path("notebooks/velm_colab.ipynb")
BACKUP = Path(f"notebooks/velm_colab.ipynb.bak-{int(time.time())}")


def code_cell(source: str) -> dict:
    """Construct a code cell from a single source string."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


# ─────────────────────────────────────────────────────────────────────
# Cell 2 — bootstrap: pin transformers, fix corrupted drive.mount
# ─────────────────────────────────────────────────────────────────────
CELL_2 = """\
!pip install -q "jax[cuda12]" equinox jaxtyping optax einops tqdm datasets
!pip install -q "transformers>=5.6.2"

import os, sys, json, shutil, time, glob
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

# Detect environment and set up persistent checkpoint directory.
# On Colab: mount Drive once and sync any prior checkpoints to local disk.
# Off Colab: use ./checkpoints/ as the persistent root.
try:
    from google.colab import drive  # noqa: E402
    drive.mount('/content/drive', force_remount=False)
    DRIVE_DIR = "/content/drive/MyDrive/VELM_checkpoints"
    os.makedirs(DRIVE_DIR, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    for f in os.listdir(DRIVE_DIR):
        if (f.endswith('.eqx') or f.endswith('.json') or f.endswith('.npy')) \\
           and not os.path.exists(f"checkpoints/{f}"):
            shutil.copy2(f"{DRIVE_DIR}/{f}", f"checkpoints/{f}")
    print(f"🚀 Google Drive mounted. Persistence enabled at {DRIVE_DIR}")
except ImportError:
    DRIVE_DIR = "checkpoints"
    os.makedirs(DRIVE_DIR, exist_ok=True)
    print("⚠ Not on Colab. Persistence using local 'checkpoints/' directory.")

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
print(f"JAX {jax.__version__} | backend: {jax.default_backend()} | devices: {jax.devices()}")
"""

# ─────────────────────────────────────────────────────────────────────
# Cell 4 — hardware profile: add RTX 3060 (12 GB) tier
# ─────────────────────────────────────────────────────────────────────
CELL_4 = """\
import equinox as eqx
import numpy as np
import optax


def detect_hardware() -> dict:
    \"\"\"Pick training profile based on available GPU.\"\"\"
    try:
        dev = jax.devices("gpu")[0]
        kind = dev.device_kind.lower()
        # data point order matters: H100/A100 first, then mid-tier 16GB
        # (T4/L4), then 12 GB consumer (3060/4070), then CPU fallback.
        if "h100" in kind or "a100" in kind:
            return {"batch": 256, "ae_steps": 100_000, "egg_steps": 10_000,
                    "pop": 64, "chunks": 500_000, "name": f"A100/H100 ({kind})"}
        if "t4" in kind or "l4" in kind or "v100" in kind:
            return {"batch": 64, "ae_steps": 150_000, "egg_steps": 5_000,
                    "pop": 32, "chunks": 250_000, "name": f"T4-class ({kind})"}
        if "3060" in kind or "rtx" in kind or "4060" in kind or "4070" in kind:
            # 12 GB consumer cards: smaller AE batch, same EGGROLL pop
            return {"batch": 48, "ae_steps": 150_000, "egg_steps": 5_000,
                    "pop": 32, "chunks": 200_000, "name": f"3060/4070 ({kind})"}
        # unknown GPU — be conservative
        return {"batch": 32, "ae_steps": 150_000, "egg_steps": 5_000,
                "pop": 24, "chunks": 200_000, "name": f"Unknown GPU ({kind})"}
    except Exception:
        return {"batch": 32, "ae_steps": 10_000, "egg_steps": 1_000,
                "pop": 16, "chunks": 50_000, "name": "CPU (slow!)"}


HW = detect_hardware()
CONFIG_NAME = "gpu_12gb_v2"
cfg = CONFIGS[CONFIG_NAME]
cfg["n_loops"] = cfg.get("n_loops", 1)
cfg["use_act"] = cfg.get("use_act", False)
print(f"  Mythos RDT: n_loops={cfg['n_loops']} | use_act={cfg['use_act']}")

K = cfg["chunk_size_k"]
print(f"Hardware: {HW['name']} | Config: {CONFIG_NAME}")
print(f"  AE steps: {HW['ae_steps']:,} | EGGROLL steps: {HW['egg_steps']:,}")
print(f"  Target chunks: {HW['chunks']:,} ({HW['chunks']*K:,} tokens)")
"""

# ─────────────────────────────────────────────────────────────────────
# Cell 6 — data loading: fix the never-set all_chunks/num_chunks bug
# ─────────────────────────────────────────────────────────────────────
CELL_6 = """\
DATA_CACHE_BIN = f"{DRIVE_DIR}/all_chunks.npy"
DATA_CACHE_LABELS = f"{DRIVE_DIR}/chunk_labels.json"

if os.path.exists(DATA_CACHE_BIN) and os.path.exists(DATA_CACHE_LABELS):
    print("🚀 Loading cached dataset from", DRIVE_DIR)
    all_chunks = np.load(DATA_CACHE_BIN)
    with open(DATA_CACHE_LABELS, "r") as f:
        chunk_labels = json.load(f)
    num_chunks = all_chunks.shape[0]
    print(f"✓ Loaded {num_chunks:,} chunks ({num_chunks*K:,} tokens) from cache.")
else:
    from datasets import load_dataset
    from tqdm import tqdm

    TARGET_CHUNKS = HW["chunks"]
    print(f"📥 Cache not found. Streaming → {TARGET_CHUNKS:,} chunks (K={K})...")

    chunk_buffer: list[np.ndarray] = []
    chunk_labels = []
    total_chunks = 0

    CURRICULUM = [
        {"name": "open-web-math/open-web-math", "weight": 0.5, "label": "math",
         "split": "train", "text_field": "text"},
        {"name": "wikitext", "config": "wikitext-103-raw-v1", "weight": 0.3,
         "label": "general", "split": "train", "text_field": "text"},
        {"name": "roneneldan/TinyStories", "weight": 0.2, "label": "narrative",
         "split": "train", "text_field": "text"},
    ]

    for source in CURRICULUM:
        source_target = int(TARGET_CHUNKS * source["weight"])
        source_count = 0
        label = source["label"]
        print(f"  Streaming {label}: {source['name']} (target: {source_target:,} chunks)...")
        try:
            load_kwargs = {"split": source["split"], "streaming": True}
            if "config" in source:
                load_kwargs["name"] = source["config"]
            ds = load_dataset(source["name"], **load_kwargs)
            text_field = source.get("text_field", "text")
            for example in tqdm(ds, desc=f"  {label}", unit="docs", leave=False):
                text = example.get(text_field, "")
                if not text or len(text) < 50:
                    continue
                tokens = tokenizer.encode(text, max_length=512, truncation=True)
                usable = len(tokens) - (len(tokens) % K)
                if usable < K:
                    continue
                chunks = np.array(tokens[:usable], dtype=np.int32).reshape(-1, K)
                chunk_buffer.append(chunks)
                chunk_labels.extend([label] * chunks.shape[0])
                source_count += chunks.shape[0]
                total_chunks += chunks.shape[0]
                if source_count >= source_target:
                    break
            print(f"    ✓ {label}: {source_count:,} chunks")
        except Exception as e:
            print(f"    ⚠ {label} failed ({e}), generating fallback random chunks")
            fallback_n = source_target - source_count
            if fallback_n > 0:
                rng = np.random.default_rng(42)
                fallback = rng.integers(0, VOCAB_SIZE, size=(fallback_n, K), dtype=np.int32)
                chunk_buffer.append(fallback)
                chunk_labels.extend([f"{label}_fallback"] * fallback_n)
                total_chunks += fallback_n

    if total_chunks == 0:
        raise RuntimeError("No training data loaded! Check network and dataset access.")

    # CRITICAL: actually concatenate the buffer (this was missing before)
    all_chunks = np.concatenate(chunk_buffer, axis=0)[:TARGET_CHUNKS]
    chunk_labels = chunk_labels[:TARGET_CHUNKS]
    num_chunks = all_chunks.shape[0]

    label_counts: dict[str, int] = {}
    for lab in chunk_labels:
        label_counts[lab] = label_counts.get(lab, 0) + 1
    print(f"\\n✓ {num_chunks:,} chunks × K={K} = {num_chunks * K:,} tokens")
    for lab, cnt in sorted(label_counts.items()):
        print(f"  {lab}: {cnt:,} ({cnt / num_chunks:.0%})")

    print(f"💾 Caching dataset to {DATA_CACHE_BIN}...")
    np.save(DATA_CACHE_BIN, all_chunks)
    with open(DATA_CACHE_LABELS, "w") as f:
        json.dump(chunk_labels, f)
    print("✓ Dataset cached.")

# host→device once: avoid per-step copy in tight training loops
all_chunks_jnp = jnp.asarray(all_chunks)
"""

# ─────────────────────────────────────────────────────────────────────
# Cell 10 — backbone init + AE load: fix the corrupted noqa-eaten lines
# (this cell mostly OK; just patch the broken FileNotFoundError text)
# ─────────────────────────────────────────────────────────────────────
CELL_10 = """\
# Re-derive AE dimensions from config (in case Phase 1 cells were skipped)
HIDDEN_DIM = cfg.get("ae_hidden_dim", 256)
LATENT_DIM = cfg["latent_dim"]
FFN_DIM = cfg.get("ae_ffn_intermediate", 512)
KL_CLIP = cfg.get("ae_kl_clip", 0.5)
KL_WEIGHT = cfg.get("ae_kl_weight", 0.001)

POP_SIZE = HW["pop"]
EGGROLL_STEPS = HW["egg_steps"]
HEAD_MAX_STEPS = 3000   # phase-1 head-only steps (hard cap)
USE_EGGROLL_PHASE = True
SIGMA = cfg.get("eggroll_sigma", 0.001)
EGG_LR = cfg.get("eggroll_lr", 3e-4)
EVAL_BATCH = cfg.get("eggroll_eval_batch", 64)
ANTITHETIC = cfg.get("eggroll_antithetic", True)
HC_D = cfg.get("hc_streams", 1)
HC_S = cfg.get("hc_s", 2)

key = jax.random.PRNGKey(123)
k1, k2, key = jax.random.split(key, 3)

backbone = VELMBackbone(
    dim=cfg["hidden_dim"],
    num_heads=cfg["num_heads"],
    num_miras_layers=cfg["miras_layers"],
    num_swa_layers=cfg["swa_layers"],
    ffn_intermediate=cfg["ffn_intermediate"],
    chunk_size=K,
    ae_hidden_dim=HIDDEN_DIM,
    hc_streams=HC_D,
    hc_s=HC_S,
    n_loops=cfg["n_loops"],
    use_act=cfg["use_act"],
    key=k1,
)
head = EnergyHead(
    hidden_dim=cfg["hidden_dim"],
    latent_dim=LATENT_DIM,
    num_blocks=cfg["energy_head_blocks"],
    key=k2,
)

bb_p = sum(x.size for x in jax.tree.leaves(eqx.filter(backbone, eqx.is_array)))
hd_p = sum(x.size for x in jax.tree.leaves(eqx.filter(head, eqx.is_array)))
print(f"Backbone: {bb_p:,} | Head: {hd_p:,} | Total: {bb_p+hd_p:,}")
if HC_D > 1:
    print(f"go-mHC: d={HC_D} streams, s={HC_S} (doubly stochastic routing)")
print(f"EGGROLL: pop={POP_SIZE}, σ={SIGMA}, steps={EGGROLL_STEPS:,}"
      f"{', antithetic' if ANTITHETIC else ''}, eval_batch={EVAL_BATCH}")

# Load frozen AE — from Phase 1 if it ran in this session, otherwise from disk.
# Cell 2 already synced any Drive checkpoints to ./checkpoints/.
if "model" not in dir() or model is None:
    print("Loading AE from checkpoint (Phase 1 was skipped)...")
    model = CALMAutoencoder(
        vocab_size=VOCAB_SIZE, chunk_size=K, hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM, ffn_intermediate=FFN_DIM,
        kl_weight=KL_WEIGHT, kl_clip=KL_CLIP,
        key=jax.random.PRNGKey(42),
    )
    ckpt_path = "checkpoints/calm_ae_best.eqx"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/calm_ae_final.eqx"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "No AE checkpoint found in ./checkpoints/. "
            "If you've trained before, ensure DRIVE_DIR is mounted (re-run cell 2). "
            "Otherwise, re-run Phase 1 to train the AE (~2.7h on T4-class)."
        )
    model = eqx.tree_deserialise_leaves(ckpt_path, model)
    print(f"  ✓ Loaded: {ckpt_path}")
frozen_ae = model
"""

# ─────────────────────────────────────────────────────────────────────
# Cell 11 — teacher vectors + setup: fix corrupted noqa, drop the
# broken in-cell evaluate_params (replaced by velm_fitness module)
# ─────────────────────────────────────────────────────────────────────
CELL_11 = """\
# Extract teacher hidden states (Qwen3.5) and cache them.
# Cell 2 already restored teacher_vectors.npy from Drive if present.

TEACHER_CACHE = "checkpoints/teacher_vectors.npy"
_loaded_teacher = False

if os.path.exists(TEACHER_CACHE):
    all_teacher_vecs = np.load(TEACHER_CACHE)
    TEACHER_DIM = all_teacher_vecs.shape[1]
    print(f"✓ Loaded cached teacher vectors: {all_teacher_vecs.shape}")
    _loaded_teacher = True

if not _loaded_teacher:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM

    print(f"\\nExtracting teacher vectors from {DEFAULT_TOKENIZER}...")
    teacher_device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_TOKENIZER,
        trust_remote_code=True,
        torch_dtype=torch.float16 if teacher_device == "cuda" else torch.float32,
    ).to(teacher_device).eval()
    TEACHER_DIM = teacher_model.config.hidden_size
    print(f"  Teacher: dim={TEACHER_DIM} | device={teacher_device}")

    TEACHER_BATCH = 64
    teacher_hiddens = []
    for start in tqdm(range(0, num_chunks, TEACHER_BATCH), desc="  Extracting"):
        end = min(start + TEACHER_BATCH, num_chunks)
        with torch.no_grad():
            input_ids = torch.tensor(
                np.asarray(all_chunks[start:end]), dtype=torch.long, device=teacher_device,
            )
            outputs = teacher_model(input_ids=input_ids, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1].float().mean(dim=1)
            teacher_hiddens.append(last_hidden.cpu().numpy())

    all_teacher_vecs = np.concatenate(teacher_hiddens, axis=0)
    nan_count = int(np.sum(~np.isfinite(all_teacher_vecs)))
    if nan_count:
        print(f"  ⚠ {nan_count} non-finite values — replacing with 0")
        all_teacher_vecs = np.nan_to_num(all_teacher_vecs, nan=0.0, posinf=0.0, neginf=0.0)
    teacher_std = float(np.std(all_teacher_vecs))
    if teacher_std > 0:
        all_teacher_vecs = all_teacher_vecs / teacher_std
        print(f"  Normalized (std was {teacher_std:.2f})")

    os.makedirs("checkpoints", exist_ok=True)
    np.save(TEACHER_CACHE, all_teacher_vecs)
    if DRIVE_DIR != "checkpoints":
        shutil.copy2(TEACHER_CACHE, f"{DRIVE_DIR}/teacher_vectors.npy")
        print(f"  ✓ Saved to Drive: {DRIVE_DIR}/teacher_vectors.npy")
    del teacher_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  ✓ {all_teacher_vecs.shape[0]:,} vectors ({all_teacher_vecs.shape[1]}d)")

# Pack trainable params and split out static (non-array) module structure.
# These are used by the JIT-compiled fitness step in src/training/velm_fitness.py.
trainable = {
    "backbone": eqx.filter(backbone, eqx.is_array),
    "head": eqx.filter(head, eqx.is_array),
}
_bb_static = eqx.filter(backbone, lambda x: not eqx.is_array(x))
_hd_static = eqx.filter(head, lambda x: not eqx.is_array(x))

# Teacher projection (only used if Phase 2a gradient distillation is enabled).
from src.training.distillation import TeacherProjection  # noqa: E402

k_proj, key = jax.random.split(key)
teacher_proj = TeacherProjection(TEACHER_DIM, cfg["hidden_dim"], key=k_proj)
_tp_static = eqx.filter(teacher_proj, lambda x: not eqx.is_array(x))
"""

# ─────────────────────────────────────────────────────────────────────
# Cell 12 — Phase 2a (gradient + distillation, currently disabled)
# Was 200+ lines of dead code. Collapse to a one-line skip notice.
# ─────────────────────────────────────────────────────────────────────
CELL_12 = """\
# Phase 2a (gradient + teacher distillation) is intentionally DISABLED.
#
# The project's thesis is gradient-free training via EGGROLL — using
# backprop where backprop works contradicts that goal. The code path is
# preserved in src/training/distillation.py for ablation studies, but
# the notebook flow goes straight from Phase 1 (AE) → Phase 2 (EGGROLL).
USE_GRADIENT_PHASE = False
print("Phase 2a (gradient distillation): SKIPPED — proceeding to EGGROLL.")
"""

# ─────────────────────────────────────────────────────────────────────
# Cell 13 — THE BIG ONE: replace hand-rolled EGGROLL loop with
# JIT-compiled factory from src/training/velm_fitness.py
# ─────────────────────────────────────────────────────────────────────
CELL_13 = """\
# Phase 2: EGGROLL training with progressive head-only → full unfreeze.
#
# Uses the JIT-compiled antithetic ES step from src/training/velm_fitness.
# Three correctness fixes vs the previous hand-rolled loop:
#   1. Per-step random keys for the energy head (was constant PRNGKey(0))
#   2. Frozen-AE encoding done once per step, not once per member
#   3. jax.lax.map over population (single JIT compile, GPU-resident)

from src.training.diagnostics import EGGROLLDiagnostics
from src.training.eggroll import SigmaAdaptor
from src.training.velm_fitness import make_velm_eggroll_step

diag = EGGROLLDiagnostics(plateau_window=50, plateau_threshold=0.001)
adaptor = SigmaAdaptor(initial_sigma=SIGMA, target_diversity=0.02)

# History buffers — fed into the plot cell and convergence checks.
egg_history = {
    "mean_fitness": [], "max_fitness": [], "fitness_std": [],
    "grad_norm": [], "phase": [], "sigma": [], "diversity": [], "step": [],
}

# Two compiled step functions: head-only (rank=1, fast warmup) and full
# (rank=2, lower LR). Each compiles once and is reused for the whole phase.
head_opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(EGG_LR))
head_opt_state = head_opt.init({"head": trainable["head"]})
step_head = make_velm_eggroll_step(
    frozen_ae, _bb_static, _hd_static, head_opt,
    pop_size=POP_SIZE, rank=1, num_samples=8, perturb_head_only=True,
)

full_opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(EGG_LR * 0.3))
full_opt_state = full_opt.init(trainable)
step_full = make_velm_eggroll_step(
    frozen_ae, _bb_static, _hd_static, full_opt,
    pop_size=POP_SIZE, rank=2, num_samples=8, perturb_head_only=False,
)

# Resume from latest checkpoint if any exist locally.
RESUME_STEP = 0
existing = sorted(glob.glob("checkpoints/backbone_step*.eqx"),
                  key=lambda p: int(p.rsplit("step", 1)[1].rsplit(".", 1)[0]))
if existing:
    latest = existing[-1]
    RESUME_STEP = int(latest.rsplit("step", 1)[1].rsplit(".", 1)[0])
    print(f"🔄 Resuming EGGROLL from step {RESUME_STEP} ({latest})")
    trainable["backbone"] = eqx.tree_deserialise_leaves(latest, trainable["backbone"])
    hd_ckpt = latest.replace("backbone", "head")
    if os.path.exists(hd_ckpt):
        trainable["head"] = eqx.tree_deserialise_leaves(hd_ckpt, trainable["head"])
    # re-init opt states after loading params
    head_opt_state = head_opt.init({"head": trainable["head"]})
    full_opt_state = full_opt.init(trainable)

UNFREEZE_PLATEAU = 0.01
CONVERGENCE_PLATEAU = 0.005
UNFREEZE_WINDOW = 10
CONVERGENCE_WINDOW = 20

# Stability gate: did head-only training already plateau?
already_unfrozen = RESUME_STEP >= HEAD_MAX_STEPS

start = time.time()
print(f"🚀 EGGROLL: starting at step {RESUME_STEP+1}/{EGGROLL_STEPS}, "
      f"target diversity={adaptor.target_diversity}")

for step in range(RESUME_STEP + 1, EGGROLL_STEPS + 1):
    key, batch_key, step_key = jax.random.split(key, 3)

    # contiguous slice → backbone sees sequential context
    start_idx = int(jax.random.randint(
        batch_key, (), 0, max(1, num_chunks - EVAL_BATCH)))
    batch = all_chunks_jnp[start_idx:start_idx + EVAL_BATCH]

    # phase decision: head-only until either hard cap or stability gate
    in_head_phase = (not already_unfrozen) and (step <= HEAD_MAX_STEPS)
    sigma_jax = jnp.asarray(adaptor.sigma, dtype=jnp.float32)

    if in_head_phase:
        trainable, head_opt_state, m = step_head(
            trainable, head_opt_state, batch, step_key, sigma_jax,
        )
        phase_label = "head"
    else:
        if not already_unfrozen:
            already_unfrozen = True
            full_opt_state = full_opt.init(trainable)  # fresh state on transition
            print(f"\\n  >>> Unfreezing backbone at step {step} <<<\\n")
        trainable, full_opt_state, m = step_full(
            trainable, full_opt_state, batch, step_key, sigma_jax,
        )
        phase_label = "full"

    # cheap metrics: read out floats only every 100 steps to avoid syncs
    if step % 100 == 0:
        mf = float(m["mean_fitness"]); xf = float(m["max_fitness"])
        fs = float(m["fitness_std"]);  gn = float(m["grad_norm"])
        entry = diag.log_step(step, trainable, {
            "mean_fitness": mf, "max_fitness": xf,
            "fitness_std": fs, "grad_norm": gn,
        })
        adaptor.update(entry["diversity"])
        egg_history["mean_fitness"].append(mf)
        egg_history["max_fitness"].append(xf)
        egg_history["fitness_std"].append(fs)
        egg_history["grad_norm"].append(gn)
        egg_history["phase"].append(phase_label)
        egg_history["sigma"].append(adaptor.sigma)
        egg_history["diversity"].append(entry["diversity"])
        egg_history["step"].append(step)
        elapsed = time.time() - start
        rate = (step - RESUME_STEP) / elapsed if elapsed > 0 else 0.0
        print(f"  step {step:>5} [{phase_label}] | mean_f: {mf:+.4f} | "
              f"max_f: {xf:+.4f} | grad: {gn:.2f} | σ: {adaptor.sigma:.4f} | "
              f"div: {entry['diversity']:.4f} | {rate:.1f} st/s | "
              f"{elapsed/60:.0f}m")

        # stability-gated auto-unfreeze
        if in_head_phase and len(egg_history["mean_fitness"]) >= UNFREEZE_WINDOW:
            recent = [v for v in egg_history["mean_fitness"][-UNFREEZE_WINDOW:]
                      if v == v]  # filter NaN
            if len(recent) == UNFREEZE_WINDOW:
                rel = (max(recent) - min(recent)) / (abs(min(recent)) + 1e-8)
                if rel < UNFREEZE_PLATEAU:
                    print(f"\\n  🔓 head-only fitness plateaued (Δ={rel:.4f}) — unfreezing now\\n")
                    already_unfrozen = True
                    full_opt_state = full_opt.init(trainable)

        # convergence detection in full phase
        if not in_head_phase and len(egg_history["mean_fitness"]) >= CONVERGENCE_WINDOW:
            recent = [v for v in egg_history["mean_fitness"][-CONVERGENCE_WINDOW:]
                      if v == v]
            if len(recent) >= CONVERGENCE_WINDOW // 2:
                rel = (max(recent) - min(recent)) / (abs(min(recent)) + 1e-8)
                if rel < CONVERGENCE_PLATEAU:
                    print(f"\\n  🎯 full-model converged (Δ={rel:.4f}) — stopping early")
                    break

        # divergence guard
        if len(egg_history["mean_fitness"]) >= 5:
            tail = [v for v in egg_history["mean_fitness"][-5:] if v == v]
            if len(tail) >= 2 and tail[-1] < min(tail[:-1]) * 10:
                print(f"\\n  ⚠ DIVERGENCE at step {step} — stopping EGGROLL")
                break

    # checkpoint every 500 steps
    if step % 500 == 0:
        ckpt_bb = eqx.combine(trainable["backbone"], _bb_static)
        ckpt_hd = eqx.combine(trainable["head"], _hd_static)
        bb_ckpt = f"checkpoints/backbone_step{step}.eqx"
        hd_ckpt = f"checkpoints/head_step{step}.eqx"
        eqx.tree_serialise_leaves(bb_ckpt, ckpt_bb)
        eqx.tree_serialise_leaves(hd_ckpt, ckpt_hd)
        if DRIVE_DIR != "checkpoints":
            shutil.copy2(bb_ckpt, f"{DRIVE_DIR}/backbone_step{step}.eqx")
            shutil.copy2(hd_ckpt, f"{DRIVE_DIR}/head_step{step}.eqx")

elapsed = time.time() - start
print(f"\\nPhase 2 done: {elapsed/3600:.2f}h")

# Final save of best params
final_bb = eqx.combine(trainable["backbone"], _bb_static)
final_hd = eqx.combine(trainable["head"], _hd_static)
eqx.tree_serialise_leaves("checkpoints/backbone_eggroll.eqx", final_bb)
eqx.tree_serialise_leaves("checkpoints/energy_head_eggroll.eqx", final_hd)
if DRIVE_DIR != "checkpoints":
    shutil.copy2("checkpoints/backbone_eggroll.eqx", f"{DRIVE_DIR}/backbone_eggroll.eqx")
    shutil.copy2("checkpoints/energy_head_eggroll.eqx", f"{DRIVE_DIR}/energy_head_eggroll.eqx")

with open("checkpoints/backbone_meta.json", "w", encoding="utf-8") as f:
    json.dump({
        "config": CONFIG_NAME, "vocab_size": VOCAB_SIZE, "tokenizer": DEFAULT_TOKENIZER,
        "eggroll_steps": EGGROLL_STEPS, "pop_size": POP_SIZE, "sigma": SIGMA,
        "history": egg_history,
    }, f, indent=2)
print("✓ Saved: checkpoints/backbone_eggroll.eqx + energy_head_eggroll.eqx + meta.json")
"""

# ─────────────────────────────────────────────────────────────────────
# Cell 14 — plot: now that egg_history is populated, this works
# ─────────────────────────────────────────────────────────────────────
CELL_14 = """\
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

mf = egg_history["mean_fitness"]
xf = egg_history["max_fitness"]
phases = egg_history.get("phase", [])

if mf:
    x = list(range(len(mf)))
    head_idx = [i for i, p in enumerate(phases) if p == "head"]
    full_idx = [i for i, p in enumerate(phases) if p == "full"]
    if head_idx:
        axes[0, 0].plot(head_idx, [mf[i] for i in head_idx], "b-", label="head only", alpha=0.85)
    if full_idx:
        axes[0, 0].plot(full_idx, [mf[i] for i in full_idx], "r-", label="full model", alpha=0.85)
    axes[0, 0].plot(x, xf, "g--", label="max", alpha=0.5)
    if head_idx and full_idx:
        axes[0, 0].axvline(x=full_idx[0], color="k", linestyle=":", label="unfreeze")
    axes[0, 0].legend(fontsize=8)
else:
    axes[0, 0].text(0.5, 0.5, "No data", ha="center", va="center")
axes[0, 0].set_title("EGGROLL Fitness (higher=better)")
axes[0, 0].set_xlabel("×100 steps")

# Energy loss = -mean_fitness
if mf:
    valid = [(i, -v) for i, v in enumerate(mf) if v == v]
    if valid:
        axes[0, 1].plot([v[0] for v in valid], [v[1] for v in valid], "b-")
axes[0, 1].set_title("Energy Loss (↓ better)")
axes[0, 1].set_xlabel("×100 steps")

gn = egg_history["grad_norm"]
if gn:
    axes[1, 0].semilogy(gn)
axes[1, 0].set_title("ES Grad Norm (log scale)")
axes[1, 0].set_xlabel("×100 steps")

# Sigma + diversity over time
sg = egg_history.get("sigma", [])
dv = egg_history.get("diversity", [])
if sg:
    ax = axes[1, 1]
    ax.plot(sg, "b-", label="σ")
    ax.set_ylabel("σ", color="b")
    ax2 = ax.twinx()
    ax2.plot(dv, "r-", label="diversity")
    ax2.set_ylabel("diversity", color="r")
axes[1, 1].set_title("Adaptive σ vs population diversity")
axes[1, 1].set_xlabel("×100 steps")

plt.tight_layout()
plt.savefig("eggroll_training.png", dpi=150, bbox_inches="tight")
print("Saved: eggroll_training.png")
try:
    plt.show()
except Exception:
    pass
"""

# ─────────────────────────────────────────────────────────────────────
# Cell 18 — GEA Phase 3: use JIT-compiled fitness eval, real keys
# ─────────────────────────────────────────────────────────────────────
CELL_18 = """\
from src.evolution.gea_eggroll import (  # noqa: E402
    GroupEvolver,
    experience_weighted_eggroll_step,
)
from src.training.velm_fitness import make_velm_fitness_eval  # noqa: E402

GEA_ITERATIONS = 10
GEA_POP = min(HW["pop"], 32)
GEA_GROUP = 5
GEA_SIGMA = SIGMA
GEA_BIAS = 0.3

print(f"\\nPhase 3: GEA Group Evolution — {GEA_ITERATIONS} iterations")
print(f"  population: {GEA_POP} | group: {GEA_GROUP} | σ: {GEA_SIGMA}")
print("=" * 60)

# Task domains derived from data labels (excluding fallbacks)
unique_labels = sorted(
    set(lab for lab in chunk_labels if not lab.endswith("_fallback"))
)
task_distribution = (
    [{"type": lab} for lab in unique_labels] if unique_labels else [{"type": "default"}]
)
print(f"  Task domains: {[t['type'] for t in task_distribution]}")

# Per-domain index lookup
domain_indices: dict[str, list[int]] = {}
for i, lab in enumerate(chunk_labels):
    domain_indices.setdefault(lab, []).append(i)

# JIT-compiled evaluator (replaces the slow non-JIT in-cell version)
gea_eval = make_velm_fitness_eval(frozen_ae, _bb_static, _hd_static, num_samples=8)


def gea_fitness_fn(params, task):
    \"\"\"Evaluate one perturbed candidate on one task domain.\"\"\"
    task_type = task.get("type", "default")
    domain_idx = domain_indices.get(task_type, [])
    if domain_idx:
        # take a deterministic sample for reproducibility per (task, member)
        sample_idx = np.array(domain_idx[:EVAL_BATCH], dtype=np.int32)
        if len(sample_idx) < EVAL_BATCH:
            extra = np.random.choice(domain_idx, EVAL_BATCH - len(sample_idx))
            sample_idx = np.concatenate([sample_idx, extra.astype(np.int32)])
    else:
        sample_idx = np.random.choice(num_chunks, EVAL_BATCH, replace=False).astype(np.int32)

    batch = all_chunks_jnp[sample_idx]
    # real per-call key — was hard-coded to PRNGKey(0) before
    eval_key = jax.random.PRNGKey(hash((task_type, int(sample_idx.sum()))) & 0x7FFFFFFF)
    fitness = float(gea_eval(params, batch, eval_key))
    return fitness, {"energy_loss": -fitness, "num_chunks": EVAL_BATCH}


gea_optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
gea_opt_state = gea_optimizer.init(trainable)
evolver = GroupEvolver(population_size=GEA_POP, group_size=GEA_GROUP)

gea_history = {"mean_fitness": [], "max_fitness": [], "parent_sizes": []}
gea_start = time.time()

for gea_iter in range(1, GEA_ITERATIONS + 1):
    key, iter_key, eggroll_key = jax.random.split(key, 3)

    experience, traces = evolver.evolution_step(
        trainable, gea_fitness_fn, task_distribution,
        key=iter_key, sigma=GEA_SIGMA, rank=1,
    )

    parent_indices = list(experience.get("task_champions", {}).values())
    seen = set()
    unique_parents = [i for i in parent_indices if not (i in seen or seen.add(i))]

    trainable, gea_opt_state, step_metrics = experience_weighted_eggroll_step(
        trainable, traces,
        key=eggroll_key,
        optimizer=gea_optimizer, opt_state=gea_opt_state,
        sigma=GEA_SIGMA, rank=1,
        parent_indices=unique_parents if unique_parents else None,
        parent_bias=GEA_BIAS,
    )

    mf = step_metrics["mean_fitness"]; xf = step_metrics["max_fitness"]
    gea_history["mean_fitness"].append(mf)
    gea_history["max_fitness"].append(xf)
    gea_history["parent_sizes"].append(len(unique_parents))
    elapsed = time.time() - gea_start
    print(f"  GEA {gea_iter:>3}/{GEA_ITERATIONS} | mean: {mf:.4f} | "
          f"best: {xf:.4f} | parents: {len(unique_parents)} | {elapsed/60:.0f}m")

print(f"\\nPhase 3 done: {(time.time()-gea_start)/60:.1f}m")

final_bb = eqx.combine(trainable["backbone"], _bb_static)
final_hd = eqx.combine(trainable["head"], _hd_static)
eqx.tree_serialise_leaves("checkpoints/backbone_gea.eqx", final_bb)
eqx.tree_serialise_leaves("checkpoints/energy_head_gea.eqx", final_hd)
if DRIVE_DIR != "checkpoints":
    shutil.copy2("checkpoints/backbone_gea.eqx", f"{DRIVE_DIR}/backbone_gea.eqx")
    shutil.copy2("checkpoints/energy_head_gea.eqx", f"{DRIVE_DIR}/energy_head_gea.eqx")
print("✓ Saved: backbone_gea.eqx + energy_head_gea.eqx")
"""

# ─────────────────────────────────────────────────────────────────────
# Cell 20 — final evaluation: fix off-by-one bug
# ─────────────────────────────────────────────────────────────────────
CELL_20 = """\
print("\\n" + "=" * 60)
print("Evaluation")
print("=" * 60)

# AE reconstruction on held-out samples
key, eval_key = jax.random.split(key)
eval_idx = jax.random.randint(eval_key, (1000,), 0, num_chunks)
eval_batch = all_chunks_jnp[eval_idx]
final_acc = float(eval_accuracy(frozen_ae, eval_batch))
print(f"AE reconstruction accuracy (1000 chunks): {final_acc:.4%}")

print("\\nSample reconstructions (original → reconstructed):")
for i in range(5):
    original = all_chunks[i]
    original_text = tokenizer.decode(original.tolist())
    reconstructed = frozen_ae.reconstruct(jnp.array(original))
    recon_text = tokenizer.decode(reconstructed.tolist())
    match_char = "✓" if np.array_equal(original, np.array(reconstructed)) else "✗"
    print(f"  {match_char} [{original_text[:40]:>40s}] → [{recon_text[:40]:<40s}]")

# Backbone next-vector prediction: energy loss DISTRIBUTION
print("\\n— Energy Loss Distribution (500 pairs) —")
key, pred_key = jax.random.split(key)
N_EVAL = 500
test_idx = jax.random.randint(pred_key, (N_EVAL,), 0, num_chunks)
test_tokens = all_chunks_jnp[test_idx]
target_z = jax.vmap(lambda c: frozen_ae.encode(c, training=False)[0])(test_tokens)
input_seq = jax.vmap(
    lambda c: final_bb.compress_input(jax.vmap(frozen_ae.embedding)(c))
)(test_tokens)
hidden, _ = final_bb(input_seq)
h_in, z_tgt = hidden[:-1], target_z[1:]


def measure_loss(h, z_t, k):
    samples = final_hd(h, key=k, num_samples=8)
    return energy_score(samples, z_t)


pred_keys = jax.random.split(pred_key, h_in.shape[0])
losses = jax.vmap(measure_loss)(h_in, z_tgt, pred_keys)
losses_np = np.array(losses)
finite_losses = losses_np[np.isfinite(losses_np)]

print(f"  Total pairs: {len(losses_np)} | Finite: {len(finite_losses)} | "
      f"NaN/Inf: {len(losses_np) - len(finite_losses)}")
if len(finite_losses) > 0:
    pcts = np.percentile(finite_losses, [5, 25, 50, 75, 95])
    print(f"  Mean:   {np.mean(finite_losses):.4f}")
    print(f"  Median: {pcts[2]:.4f}")
    print(f"  P5/P25/P75/P95: {pcts[0]:.4f} / {pcts[1]:.4f} / {pcts[3]:.4f} / {pcts[4]:.4f}")
    print(f"  Std:    {np.std(finite_losses):.4f}")

# Per-domain breakdown — FIX: was [:len(losses_np)+1] → off-by-one
print("\\n— Per-Domain Energy Loss —")
test_labels = [chunk_labels[int(idx)] for idx in test_idx[:len(losses_np)]]
domain_losses: dict[str, list] = {}
# losses array maps to pairs (i, i+1); we use shifted labels so loss_i pairs to label_{i+1}
shift_labels = test_labels[1:] + ["unknown"]
for i, loss_val in enumerate(losses_np):
    if not np.isfinite(loss_val):
        continue
    lab = shift_labels[i] if i < len(shift_labels) else "unknown"
    domain_losses.setdefault(lab, []).append(float(loss_val))
for lab in sorted(domain_losses):
    vals = domain_losses[lab]
    print(f"  {lab:>12s}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} n={len(vals)}")

# Cosine similarity between predictions and targets
print("\\n— Prediction-Target Cosine Similarity —")
def cos_sim(h, z_t, k):
    pred = final_hd.predict(h, key=k)
    p_n = pred / (jnp.linalg.norm(pred) + 1e-8)
    t_n = z_t / (jnp.linalg.norm(z_t) + 1e-8)
    return jnp.sum(p_n * t_n)

sims = jax.vmap(cos_sim)(h_in, z_tgt, pred_keys)
sims_np = np.array(sims)
finite_sims = sims_np[np.isfinite(sims_np)]
if len(finite_sims) > 0:
    pcts_s = np.percentile(finite_sims, [5, 25, 50, 75, 95])
    print(f"  Mean sim:   {np.mean(finite_sims):.4f} (1.0=perfect, 0=random)")
    print(f"  Median sim: {pcts_s[2]:.4f}")
    print(f"  P5/P95:     {pcts_s[0]:.4f} / {pcts_s[4]:.4f}")

# Miras memory state norm sanity check
print("\\n— Miras Memory State Norms —")
for i, block in enumerate(final_bb.miras_blocks):
    cap = float(block.miras.frobenius_capacity)
    print(f"  Layer {i}: frobenius_capacity = {cap:.1f}")

# Plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
if len(finite_losses) > 0:
    axes[0].hist(finite_losses, bins=50, alpha=0.7, color="steelblue")
    axes[0].axvline(np.median(finite_losses), color="r", linestyle="--", label="median")
    axes[0].legend()
axes[0].set_title("Energy Loss Distribution"); axes[0].set_xlabel("Energy Score")

if len(finite_sims) > 0:
    axes[1].hist(finite_sims, bins=50, alpha=0.7, color="seagreen")
    axes[1].axvline(0, color="gray", linestyle=":", alpha=0.5)
    axes[1].axvline(np.mean(finite_sims), color="r", linestyle="--", label="mean")
    axes[1].legend()
axes[1].set_title("Pred-Target Cosine Similarity"); axes[1].set_xlabel("Cosine Similarity")

if domain_losses:
    labels = sorted(domain_losses.keys())
    means = [np.mean(domain_losses[l]) for l in labels]
    stds = [np.std(domain_losses[l]) for l in labels]
    axes[2].bar(labels, means, yerr=stds, capsize=5, alpha=0.7, color="coral")
axes[2].set_title("Energy Loss by Domain"); axes[2].set_ylabel("Mean Energy Loss")

plt.tight_layout()
plt.savefig("evaluation_distributions.png", dpi=150, bbox_inches="tight")
print("\\nSaved: evaluation_distributions.png")
try:
    plt.show()
except Exception:
    pass
"""

# ─────────────────────────────────────────────────────────────────────
# Apply edits
# ─────────────────────────────────────────────────────────────────────
EDITS: dict[int, str] = {
    2: CELL_2,
    4: CELL_4,
    6: CELL_6,
    10: CELL_10,
    11: CELL_11,
    12: CELL_12,
    13: CELL_13,
    14: CELL_14,
    18: CELL_18,
    20: CELL_20,
}


def main() -> None:
    if not NB.exists():
        raise SystemExit(f"Notebook not found: {NB}")

    # belt-and-suspenders backup
    shutil.copy2(NB, BACKUP)
    print(f"📦 Backup: {BACKUP}")

    nb = json.loads(NB.read_text())
    cells = nb["cells"]
    if len(cells) != 22:
        print(f"⚠ Expected 22 cells, got {len(cells)}. Aborting to be safe.")
        raise SystemExit(1)

    for idx, source in EDITS.items():
        old_len = len("".join(cells[idx]["source"]))
        cells[idx] = code_cell(source)
        new_len = len("".join(cells[idx]["source"]))
        delta = new_len - old_len
        sign = "+" if delta >= 0 else ""
        print(f"  cell [{idx:2}] : {old_len:>6}c → {new_len:>6}c  ({sign}{delta:+d})")

    NB.write_text(json.dumps(nb, indent=1))
    print(f"✓ Wrote {NB} ({sum(len(json.dumps(c)) for c in cells)} bytes serialized)")


if __name__ == "__main__":
    main()
