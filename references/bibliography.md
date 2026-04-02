# VELM References

## Primary Sources (Project Knowledge)

### CALM — Continuous Autoregressive Language Models
- **Authors:** Chenze Shao, Darren Li, Fandong Meng, Jie Zhou
- **Affiliation:** WeChat AI, Tencent / Tsinghua University
- **arXiv:** 2510.27688v1
- **Code:** https://github.com/shaochenze/calm
- **Key results:** K=4 matches Transformer baselines at lower compute; >99.9% AE accuracy
- **File:** /mnt/project/2510_27688v1.pdf

### Miras — Learning to Memorize with Robust and Expressive Memory
- **Authors:** Behrouz et al.
- **arXiv:** 2504.13173
- **Key results:** Unifies RetNet/GLA/Mamba/DeltaNet/TTT under 4-choice framework
- **Variants:** Retain (ℓ2 retention), Rember (Huber loss), Memora (elastic net)
- **File:** /mnt/project/2504_13173.md

### EGGROLL — Evolution Strategies at the Hyperscale
- **Authors:** Bidipta Sarkar, Mattie Fellows, Juan Agustin Duque, et al.
- **Affiliation:** Oxford (FLAIR, WhiRL), MILA, NVIDIA
- **Code:** https://eshyperscale.github.io/
- **Key results:** 91% batch inference throughput; int8 EGG beats fp32 Transformer
  at backprop; competitive with GRPO for LLM reasoning fine-tuning
- **File:** /mnt/project/eggroll_hyperscale.pdf

### qTTT — Test-Time Training for Long-Context LLMs
- **Authors:** Rachit Bansal, Aston Zhang, Rishabh Tiwari, et al.
- **Affiliation:** Meta, Harvard/Kempner, OpenAI, UC Berkeley, UT Austin
- **arXiv:** 2512.13898v1
- **Key results:** +12.6% and +14.1% on LongBench-v2 / ZeroScrolls for Qwen3-4B;
  provably increases target-distractor margin
- **File:** /mnt/project/2512_13898v1.pdf

### CIB — Reasoning as Compression via Conditional Information Bottleneck
- **arXiv:** 2603.08462v1
- **Key results:** ~78% fewer reasoning tokens; maintains accuracy on MATH/AIME
- **File:** /mnt/project/2603_08462v1.pdf

### GEA — Group-Evolving Agents
- **Authors:** Zhaotian Weng, Antonis Antoniades, et al.
- **Affiliation:** UC Santa Barbara
- **arXiv:** 2602.04837v1
- **Key results:** 71.0% SWE-bench Verified (vs 56.7% baseline); 88.3% Polyglot;
  transfers across GPT and Claude models
- **File:** /mnt/project/2602_04837v1.pdf

## Secondary References

### Architecture Foundations
- Touvron et al. (2023). LLaMA. Transformer++ baseline.
- Gu & Dao (2024). Mamba / Mamba2. Linear state space models.
- Yang et al. (2024a). Gated DeltaNet. Gated linear recurrence.
- Sun et al. (2024). TTT. Test-time training as hidden state.
- Ren et al. (2024). Samba. Hybrid Mamba + SWA architecture.

### Evolution Strategies
- Salimans et al. (2017). ES for RL. Foundation for EGGROLL.
- Foerster (2017). Activation-free networks via float rounding error.
- Hu et al. (2022). LoRA. Low-rank adaptation (EGGROLL analog).

### Long Context
- Liu et al. (2024). Lost in the middle. Position sensitivity.
- Kamradt (2024). Needle-in-haystack. Retrieval failure diagnosis.
- Beltagy et al. (2020). Longformer. Sparse attention patterns.

### Continuous Generation
- Tschannen et al. (2023). GIVT. Gaussian mixture continuous AR.
- Li et al. (2024). Diffusion head for continuous AR.
- team et al. (2024). Large Concept Models. Sentence-level AR.

### Scaling Laws
- Kaplan et al. (2020). Neural scaling laws.
- Hoffmann et al. (2022). Chinchilla scaling.
