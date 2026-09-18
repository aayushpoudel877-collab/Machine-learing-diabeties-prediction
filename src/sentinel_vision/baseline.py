"""Classical ML baseline: frozen CNN embeddings + LinearSVC."""
import argparse, numpy as np, torch
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from .data import make_loader
from .model import build_model

def embeddings(loader,backbone,device):
    backbone.eval(); X=[]; y=[]
    with torch.no_grad():
        for x,t in loader: X.append(backbone(x.to(device)).cpu().numpy()); y.extend(t.numpy())
    return np.vstack(X),np.array(y)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",default="data/processed"); a=p.parse_args(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m=build_model(6,True); backbone=torch.nn.Sequential(*list(m.children())[:-1],torch.nn.Flatten()).to(device)
    tr,ds=make_loader(a.data+"/train",224,32,False); te,_=make_loader(a.data+"/test",224,32,False); X,y=embeddings(tr,backbone,device); Xt,yt=embeddings(te,backbone,device)
    pred=LinearSVC(class_weight="balanced").fit(X,y).predict(Xt); print(classification_report(yt,pred,target_names=ds.classes,digits=4))
if __name__=="__main__": main()
