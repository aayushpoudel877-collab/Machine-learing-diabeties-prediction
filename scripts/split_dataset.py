import argparse,random,shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",default="data/raw"); p.add_argument("--output",default="data/processed"); p.add_argument("--seed",type=int,default=42); p.add_argument("--val",type=float,default=.15); p.add_argument("--test",type=float,default=.15); a=p.parse_args(); random.seed(a.seed)
    src=Path(a.input); dst=Path(a.output)
    for cls in sorted(x.name for x in src.iterdir() if x.is_dir()):
        imgs=[x for x in (src/cls).iterdir() if x.suffix.lower() in {".jpg",".jpeg",".png",".webp"}]; random.shuffle(imgs); n=len(imgs); nv=int(n*a.val); nt=int(n*a.test)
        for split,items in {"val":imgs[:nv],"test":imgs[nv:nv+nt],"train":imgs[nv+nt:]}.items():
            out=dst/split/cls; out.mkdir(parents=True,exist_ok=True)
            for f in items: shutil.copy2(f,out/f.name)
    print(f"Dataset split complete: {dst}")
if __name__=="__main__": main()
