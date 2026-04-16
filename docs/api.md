# FedTracker-Pro API

## Core

- `src.core.config.Config`
  - Loads and stores experiment/runtime configuration.
- `src.core.fed_tracker_pro.FedTrackerPro`
  - Main orchestration class for initialization, training, ownership verification, and robustness evaluation.

## Defense

- `src.defense.watermark.cl_watermark.ContinualLearningWatermark`
- `src.defense.fingerprint.param_fingerprint.ParametricFingerprint`
- `src.defense.adaptive_allocation.AdaptiveAllocator`
- `src.defense.crypto_verification.CryptographicVerification`
- `src.defense.unlearning_guided.UnlearningGuidedRelocation`

## Attacks

- `src.attacks.fine_tuning.FineTuningAttack`
- `src.attacks.pruning.PruningAttack`
- `src.attacks.quantization.QuantizationAttack`
- `src.attacks.overwriting.OverwritingAttack`
- `src.attacks.ambiguity.AmbiguityAttack`
- `src.attacks.model_extraction.ModelExtractionAttack`

## Experiments

- `experiments.exp_baseline.build_default_attacks`
- `experiments.exp_ablation.get_ablation_groups`
- `experiments.exp_robustness.build_robustness_attacks`
- `experiments.exp_scalability.generate_client_scenarios`
