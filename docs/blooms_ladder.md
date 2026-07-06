# The Bloom Ladder: Capability Formation as a Dependency Structure

*A conceptual companion to `poc_findings.md` and `phase3_findings.md`. Everything here generalizes from things that actually happened in this repo.*

Bloom's (revised) taxonomy orders cognitive work as **remember → understand → apply → analyze → evaluate → create**. Educators use it to sequence instruction. The claim of this document is stronger and more mechanical: the ladder is a **dependency structure for capability formation in any learning system** — each rung consumes the outputs of the rung below, each rung has a distinct *failure signature* when its prerequisites are missing, and each rung requires its own *assessment with its own imposter baseline*. VELM's Phase 3 is a case study in what happens when this is violated, and what happens when it is respected.

## 1 · The ladder as general process

Restated without the classroom vocabulary:

| Rung | General form | Output consumed by next rung |
|---|---|---|
| Remember | store & retrieve faithfully | a reliable substrate of tokens/facts/codes |
| Understand | compress & predict | representations that anticipate structure |
| Apply | transfer to new instances | procedures that work off the training manifold |
| Analyze | decompose into parts & relations | structure that can be manipulated |
| Evaluate | score, judge, verify | a signal that distinguishes good from bad |
| Create | generate novel *valid* artifacts | new instances that survive evaluation |

Two properties make this a ladder rather than a list. **Dependency:** creation samples from what evaluation can score, evaluation needs analyzed structure, analysis needs predictive representations, prediction needs faithful storage. **Distinct failure signatures:** a system missing a rung doesn't fail randomly — it fails in a recognizable way. Missing *remember* → hallucination/reconstruction errors. Missing *understand* → outputs that mimic the marginal distribution of the domain (statistically plausible, contextually dead). Missing *evaluate* → confident garbage. Missing everything but *create* → fluent degeneracy.

VELM's runs produced three of these signatures on demand: the AE at 99.9% but latents unlearned = remember without understand; the "Tom was said" collapse = marginal mimicry, understand missing; the rollout loop-babble = create attempted on a 0.218-strength understand rung.

## 2 · Base training: gradient descent climbs the ladder whether you like it or not

Nobody schedules a curriculum inside pretraining, yet one always happens: **gradient descent learns in variance order** — whatever explains the most loss first. For language that means marginal token statistics, then local co-occurrence, then syntax, then semantics, then long-range structure. This is the *implicit curriculum*, and it approximately follows the ladder from the bottom (the literature's induction-head phase transitions and grokking events are rung-boundaries becoming visible).

The design consequence, which cost this project nine failed runs to learn: **objectives must be rung-pure.** Our original loss mixed a create-rung signal (the energy score, a stochastic distribution-matching objective) into the understand-rung phase (representation learning). The upper rung's gradient noise was ~100× the lower rung's signal, so the lower rung never formed — and without it, the upper rung had nothing to model. The fix was Bloom-shaped: stage 1 trains *understand* alone (deterministic MSE), stage 2 trains *evaluate/create* (energy head) on top of *frozen* understanding (`stop_gradient` — the machine equivalent of "don't relearn arithmetic while taking calculus").

A second consequence: **anisotropy is a remember-rung pathology that blocks the understand rung.** Representations optimized only for faithful storage (reconstruction) may organize geometrically in ways that make prediction impossible to *express*, even when the information is present. Diagnose the substrate before blaming the learner (our oracle probes; LatentGate's whitening — with the caveat that the right correction is task-dependent: standardization helped our regression, full whitening hurt it).

## 3 · Optimization: the ladder bounds what an optimizer can reach

Different optimizers have different *information rates*, and the rungs have different signal-to-noise. Our EGGROLL results read cleanly through this lens: at feasible populations, ES's gradient estimate carried enough information for the high-variance bottom rung (marginal shaping — it always reached the unconditional solution) but not for the low-variance understand rung (contextual signal ≈ 1% of loss). Warm-started ES refined an already-climbed ladder at 27–92% efficiency. **ES is a lower-rung and refinement optimizer; backprop is required for rung-climbing at small signal fractions** — which is exactly why the hybrid architecture scopes ES to GEA, where the *evaluate* rung (fitness) is the interface and no gradient exists.

Learning-rate schedules are rung transitions in miniature: high LR explores coarse (bottom-rung) structure; contextual gains in every successful run arrived *during decay* — consolidation. A schedule that never decays (our 48k-horizon mistake) parks the system at the bottom rung indefinitely.

## 4 · The harness: rung-matched metrics and imposter baselines

The most exportable methodological idea from Phase 3: **for every capability claim, construct the strongest system that possesses only the rungs below, and require the model to beat it.** We call these imposter baselines:

| Rung claimed | Imposter (has only lower rungs) | Our instrument |
|---|---|---|
| Remember | — | AE roundtrip accuracy |
| Understand | unconditional predictor; no-memory linear probe | centered cosine (uncond = 0); probe bar; ridge-oracle bar |
| Apply/Analyze | copy-last; marginal *sampler* | copy baseline; uncond-sampler energy |
| Evaluate | random/constant judge | (future: GEA fitness sanity checks) |
| Create | stitched retrieval (manifold snap); degenerate looper | snap rollouts; distinct-bigram ratio; external judge NLL |

Two hard-won rules. **Metrics must match the rung**: judging *create* with a truth-matching metric (cosine to the specific continuation) is invalid beyond a step or two, because valid creation *diverges* — you need truth-independent judges (plausibility, non-degeneracy). Conversely, judging *understand* without the unconditional imposter lets marginal mimicry masquerade as comprehension (our run-1 "PASS"). **A verdict is only as good as its imposters**: every false positive in this project traces to a missing baseline, and every diagnosis to adding one.

## 5 · Agents: autonomy should be gated by the evaluate rung

Map an agent stack onto the ladder: retrieval/memory = *remember*; world-model and summarization = *understand*; tool use = *apply*; planning and decomposition = *analyze*; critics, verifiers, tests = *evaluate*; autonomous action/generation = *create*. The common failure of agent design is granting create-level autonomy to systems with a weak evaluate rung — no self-verification, no imposter-checking of their own outputs. The Bloom-informed rule: **an agent's autonomy at rung N should be bounded by the demonstrated strength of its rung N−1**, especially evaluate-before-create. (This is also how the human in this project operated: every "create" step — a new training run — was gated by a new "evaluate" instrument first. The instrument-building always paid for itself.)

GEA, VELM's self-improvement phase, is this principle as architecture: populations *create* variants, a fitness function *evaluates* them, and ES — competent exactly at refinement-under-evaluation — closes the loop.

## 6 · Teaching processes: debugging is formative assessment

The two-day arc of this project was a teaching relationship, and its efficiency came from rung-diagnosis rather than retry-harder: identify *which rung* a failure lives on, then intervene there. Task produces nothing learnable → the material is wrong (v1's entropy-saturated targets). Model mimics marginals → understand is blocked; find the blocker (anisotropy, loss noise). Model understands but creation collapses → exposure bias; scaffold the practice (teacher forcing → scheduled sampling → free running is literally graduated release of responsibility). Prompting techniques map the same way: few-shot examples scaffold *remember/apply*; chain-of-thought scaffolds *analyze*; self-critique scaffolds *evaluate*.

## 7 · Documentation: a Bloom ladder for the reader

A README is a curriculum whose student is the reader (human or agent). The ladder gives it a canonical order, and most documentation fails by rung-skipping — installation commands (*apply*) before any *understand*, or API dumps (*remember*) with no *analyze* rationale:

1. **Remember** — what this is, one paragraph; install; where things live.
2. **Understand** — the core concepts and architecture; *why it's shaped this way*.
3. **Apply** — quickstart, runnable examples, common tasks.
4. **Analyze** — internals, design rationale, how the pieces compose.
5. **Evaluate** — benchmarks, limitations, failure modes, *when not to use this*.
6. **Create** — extending, contributing, building on top.

Agent-facing docs (CLAUDE.md files, skills) compress this to the rungs agents lack: apply-level recipes plus evaluate-level warnings, because the base model already carries remember/understand for common tools. VELM's own docs now approximate the ladder by accident of history — README (understand + evaluate via the empirical-status section), findings docs (analyze/evaluate), notebooks (apply) — and could be tightened deliberately.

## 8 · Where the metaphor breaks (use as lens, not law)

Honesty section. The ladder is a heuristic with real exceptions. LLMs exhibit *jagged* rung profiles — fluent creation with shaky analysis — because imitating the surface of create-level artifacts is available directly from data in a way it is not for a child; imitation can counterfeit upper rungs. (The counterfeit is exactly what imposter baselines exist to detect.) Some capabilities co-form rather than stack: our energy head's *evaluate* (scoring rule) and *create* (sampling) train jointly. And our arm-C anomaly — a backbone that learns from token inputs but not from latent inputs an MLP handles fine — fits no rung story; some failures are just mechanism bugs. Treat the ladder as the *default hypothesis* for sequencing objectives, metrics, and autonomy, and let instruments overrule it.

## 9 · Compressed design rules

1. Objectives rung-pure; stage upper-rung losses after lower-rung representations exist, and freeze what's below.
2. The optimizer must carry enough information for the target rung; ES refines, backprop climbs.
3. Every metric gets an imposter baseline from the rungs below; every create-metric must be truth-independent.
4. Diagnose failures by rung signature before changing hyperparameters.
5. Gate autonomy (and generation) by demonstrated evaluation strength.
6. Write docs down the ladder; never rung-skip past *understand*.
7. Expect counterfeits and jagged profiles; instruments beat intuitions.
