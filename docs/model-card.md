# Model Card — SentinelVision

**Intended use:** educational demonstrations and non-critical waste classification experiments.

**Classes:** cardboard, glass, metal, paper, plastic, trash.

**Training:** ImageNet-initialized ResNet18 fine-tuned on a six-class waste-image dataset supplied by the project operator.

**Metrics:** Results are intentionally not hard-coded because they depend on the exact licensed dataset and split. Run evaluation and record macro-F1, per-class precision/recall and the confusion matrix.

**Limitations:** Lighting, angle, occlusion, mixed materials and unseen categories can reduce performance. Probabilities are not guarantees.

**Safety:** Do not use predictions as the sole basis for hazardous-material handling or other safety-critical decisions.
