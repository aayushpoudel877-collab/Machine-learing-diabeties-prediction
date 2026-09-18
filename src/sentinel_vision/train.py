import argparse
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from .data import make_loader
from .model import build_model, freeze_backbone, unfreeze_all
from .utils import seed_everything

def run(cfg):
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, train_ds = make_loader("data/processed/train", cfg["image_size"], cfg["batch_size"], True, cfg["num_workers"])
    val_loader, _ = make_loader("data/processed/val", cfg["image_size"], cfg["batch_size"], False, cfg["num_workers"])
    model = build_model(cfg["num_classes"], cfg["pretrained"]).to(device)
    weights = compute_class_weight("balanced", classes=train_ds.classes, y=[y for _, y in train_ds.samples])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device), label_smoothing=cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=1, factor=0.3)
    best, stale = 0.0, 0
    freeze_backbone(model)
    for epoch in range(cfg["epochs"]):
        if epoch == cfg["freeze_backbone_epochs"]: unfreeze_all(model)
        model.train(); seen = correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(); out = model(x); loss = criterion(out, y); loss.backward(); optimizer.step()
            correct += (out.argmax(1) == y).sum().item(); seen += y.size(0)
        model.eval(); val_correct = val_seen = 0
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x.to(device)); val_correct += (out.argmax(1).cpu() == y).sum().item(); val_seen += y.size(0)
        val_acc = val_correct / max(val_seen, 1); scheduler.step(val_acc)
        print(f"epoch={epoch+1} train_acc={correct/max(seen,1):.4f} val_acc={val_acc:.4f}")
        if val_acc > best:
            best, stale = val_acc, 0; Path(cfg["checkpoint"]).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "classes": train_ds.classes, "image_size": cfg["image_size"], "val_acc": best}, cfg["checkpoint"])
        else:
            stale += 1
            if stale >= cfg["patience"]: break
    print(f"best_val_accuracy={best:.4f}; checkpoint={cfg['checkpoint']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f: run(yaml.safe_load(f))
