# SentinelVision ♻️

SentinelVision is a production-style computer-vision system for classifying everyday waste into six categories: cardboard, glass, metal, paper, plastic, and trash.

It is structured like a small ML team would build it: reproducible data pipelines, a classical ML baseline, transfer-learning CNN, evaluation, explainability-ready inference, FastAPI, Streamlit, tests, Docker, CI, and a 30-day delivery plan.

> This is built on a feature branch of the existing repository because the connected GitHub integration does not expose repository-creation permissions. The original diabetes_prediction.py is untouched.

## Stack
Python • PyTorch • Torchvision • scikit-learn • FastAPI • Streamlit • Docker • GitHub Actions

## Dataset
Use a legally usable waste-image dataset such as TrashNet or another six-class dataset. Do not commit data or model weights. Expected ImageFolder layout:

    data/raw/
      cardboard/
      glass/
      metal/
      paper/
      plastic/
      trash/

Then run:

    python scripts/download_dataset.py
    python scripts/split_dataset.py --input data/raw --output data/processed
    pip install -e ".[dev]"
    python -m sentinel_vision.train --config configs/config.yaml
    python -m sentinel_vision.evaluate --checkpoint artifacts/best.pt --data data/processed/test

## Run

API: uvicorn api.main:app --reload

UI: streamlit run app.py

API endpoint: POST /predict with multipart field file. GET /health reports model availability.

## Architecture
ImageFolder → augmentation/preprocessing → ResNet18 transfer learning → confidence/top-k → FastAPI/Streamlit.

A classical LinearSVC on frozen CNN embeddings is included as a baseline.

## Layout

    api/                    FastAPI service
    configs/                Experiment configuration
    docs/                   Architecture, model card, 30-day plan
    scripts/                Data preparation utilities
    src/sentinel_vision/    ML/DL package
    tests/                  Automated tests
    app.py                  Streamlit demo
    Dockerfile              Container image
    docker-compose.yml       Local deployment
    pyproject.toml          Dependencies/tooling

## Responsible use
This is an educational prototype. Predictions may be wrong, especially on images unlike the training distribution. Confidence is not a guarantee and the system should not be used for safety-critical disposal decisions without human verification.

See docs/30-day-plan.md and docs/model-card.md for the engineering plan and limitations.
