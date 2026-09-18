from pathlib import Path
import torch
from PIL import Image
from .data import transforms_for
from .model import build_model

class Predictor:
    def __init__(self, checkpoint="artifacts/best.pt"):
        self.path = Path(checkpoint); self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self.path.exists(): raise FileNotFoundError(f"Checkpoint not found: {self.path}")
        ckpt = torch.load(self.path, map_location=self.device, weights_only=False)
        self.classes = ckpt["classes"]; self.size = ckpt.get("image_size", 224)
        self.model = build_model(len(self.classes), pretrained=False); self.model.load_state_dict(ckpt["model"]); self.model.to(self.device).eval()
        self.transform = transforms_for(self.size, False)

    def predict(self, image: Image.Image, top_k=3):
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad(): probs = torch.softmax(self.model(x), dim=1)[0]
        values, indices = torch.topk(probs, min(top_k, len(self.classes)))
        return {"prediction": self.classes[indices[0].item()], "confidence": float(values[0]), "top_k": [{"class": self.classes[i.item()], "probability": float(v)} for v, i in zip(values, indices)]}
