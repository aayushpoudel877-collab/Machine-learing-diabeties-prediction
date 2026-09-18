"""Create the expected class folders; dataset licensing/source remains explicit."""
from pathlib import Path
CLASSES=("cardboard","glass","metal","paper","plastic","trash")
root=Path("data/raw"); root.mkdir(parents=True,exist_ok=True)
for name in CLASSES: (root/name).mkdir(exist_ok=True)
print("Created data/raw class folders. Add a licensed dataset before training.")
