import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from .data import make_loader
from .model import build_model

def evaluate(checkpoint, data_root):
    ckpt=torch.load(checkpoint,map_location="cpu",weights_only=False); classes=ckpt["classes"]
    loader,_=make_loader(data_root,ckpt.get("image_size",224),32,False,2); model=build_model(len(classes),False); model.load_state_dict(ckpt["model"]); model.eval()
    ys=[]; ps=[]
    with torch.no_grad():
        for x,y in loader: ps.extend(model(x).argmax(1).tolist()); ys.extend(y.tolist())
    print(classification_report(ys,ps,target_names=classes,digits=4)); print("macro_f1=",round(f1_score(ys,ps,average="macro"),4))
    ConfusionMatrixDisplay(confusion_matrix(ys,ps),display_labels=classes).plot(xticks_rotation=45); Path("artifacts").mkdir(exist_ok=True); plt.tight_layout(); plt.savefig("artifacts/confusion_matrix.png",dpi=160)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",default="artifacts/best.pt"); p.add_argument("--data",default="data/processed/test"); a=p.parse_args(); evaluate(a.checkpoint,a.data)
