# Architecture

## Data
ImageFolder data is kept outside Git. A deterministic split creates train/validation/test sets; augmentation is training-only.

## Models
A LinearSVC over frozen ResNet18 embeddings provides a classical baseline. The main model uses ImageNet-pretrained ResNet18, a dropout classification head, class-weighted cross entropy and label smoothing. The backbone can be frozen for an initial epoch and then fine-tuned.

## Evaluation
Track accuracy, macro-F1, per-class precision/recall and the confusion matrix. Macro-F1 prevents strong majority-class performance from hiding weak minority-class behavior.

## Serving
FastAPI and Streamlit share the Predictor class, keeping preprocessing and inference consistent.

## Next production steps
Model registry, experiment tracking, drift monitoring, privacy-aware logs, ONNX/TorchScript export, authentication/rate limiting, human feedback, and reviewed scheduled retraining.
