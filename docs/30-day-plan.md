# 30-Day Team Delivery Plan

This is a realistic four-person workflow. It describes the intended development schedule; it is not a claim that the repository was literally produced over 30 days.

| Days | Workstream | Deliverable |
|---|---|---|
| 1–2 | Product | Problem, classes, metrics |
| 3–4 | Data | Dataset audit, source/licensing notes |
| 5–6 | Data | Cleaning, duplicates, reproducible split |
| 7–8 | EDA | Distribution and image-quality analysis |
| 9–10 | ML | CNN-embedding + SVM baseline |
| 11–12 | DL | ResNet18 transfer-learning pipeline |
| 13–14 | DL | Augmentation, class weighting, smoothing |
| 15–16 | DL | Fine-tuning and checkpointing |
| 17–18 | Evaluation | Macro-F1, per-class metrics, confusion matrix |
| 19 | Explainability | Saliency/error investigation |
| 20–21 | Backend | FastAPI contract and health checks |
| 22 | UI | Streamlit demo |
| 23 | QA | Unit tests and validation |
| 24 | DevOps | Docker |
| 25 | CI | GitHub Actions |
| 26 | Robustness | Low-quality/OOD image review |
| 27 | Docs | Architecture and model card |
| 28 | Review | Error analysis and hard examples |
| 29 | Polish | Reproducibility/refactoring |
| 30 | Release | Demo rehearsal and retrospective |

### Team roles
ML engineer; data engineer; backend/ML platform engineer; frontend/QA engineer. A solo developer can rotate through these roles.
