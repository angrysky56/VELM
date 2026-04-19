# CONTEXT: Phase 1.1 - Mythos-Enhanced Architecture

## 🔍 Research Context
Based on the audit of **OpenMythos (RDT)**, the current Miras backbone lacks two critical features for stable deep reasoning:
1. **Spectral Stability**: There is no hard constraint on the recurrent weights, making $n\_loops > 1$ risky.
2. **Loop Identity**: Layers have no way of knowing if they are in the first loop or the tenth, leading to redundant computation.

## 💡 Key Decisions
- **JAX/Equinox Pattern**: We will use log-parameterization ($log\_A$ and $log\_dt$) to ensure $A = exp(-exp(...))$. This is the standard "LTI-stable" construction.
- **Injection Point**: We inject the *entire* compressed input $e$ at every step. This matches the "Anchor Injection" pattern which is superior for long-chain reasoning.
- **Backbone Loop**: The entire stack of layers (Miras + SWA) will be treated as a single recurrent unit.

## 🛠️ Resources
- Reference Implementation: `OpenMythos/open_mythos/main.py`
- Theoretical Basis: Linear Time Invariant (LTI) systems in Transformers.
