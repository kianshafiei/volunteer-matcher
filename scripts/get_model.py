"""Download all-MiniLM-L6-v2 (ONNX) from Hugging Face into ./model (about 90 MB)."""
import hashlib, json, urllib.request
from pathlib import Path

REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"   # pinned revision
BASE = f"https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/{REV}/"
OUT = Path(__file__).resolve().parent.parent / "model"
OUT.mkdir(exist_ok=True)

if not (OUT / "model.onnx").exists():
    urllib.request.urlretrieve(BASE + "onnx/model.onnx", OUT / "model.onnx")
if not (OUT / "vocab.txt").exists():
    tok = json.loads(urllib.request.urlopen(BASE + "tokenizer.json").read())
    vocab = sorted(tok["model"]["vocab"].items(), key=lambda kv: kv[1])
    (OUT / "vocab.txt").write_text("\n".join(k for k, _ in vocab) + "\n", encoding="utf-8")
print("model.onnx sha256:", hashlib.sha256((OUT / "model.onnx").read_bytes()).hexdigest()[:16])
